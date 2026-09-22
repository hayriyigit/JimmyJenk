"""System-One core: compile each question independently, score opaque labels, turn logprobs into Jev-like answers."""

import asyncio
import json
import math
import os
import time
from pathlib import Path

from app import prompts
from app.registry import ModelConfig, backend_for

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CALIBRATION_PATH = Path(os.environ.get("SYSTEMONE_CALIBRATION", Path(__file__).resolve().parent.parent / "calibration.json"))
TRACE_PATH = Path(os.environ.get("SYSTEMONE_TRACE_PATH", "data/traces/traces.jsonl"))


class UnsupportedQuestion(ValueError):
    pass


class CandidateEncoder:
    """Maps semantic options onto opaque labels that must each be exactly one token for the active tokenizer."""

    def __init__(self, options: list[str]):
        if len(options) > len(LABELS):
            raise UnsupportedQuestion(f"{len(options)} options; the single-token path supports at most {len(LABELS)}")
        self.mapping = dict(zip(LABELS, options))  # label -> option

    async def validate_single_token(self, backend) -> dict[str, int]:
        ids = await asyncio.gather(*(backend.token_ids(label) for label in self.mapping))
        bad = [label for label, t in zip(self.mapping, ids) if len(t) != 1]
        if bad:
            raise UnsupportedQuestion(f"labels {bad} are not single tokens for this tokenizer; multi-token scoring is not implemented")
        return {label: t[0] for label, t in zip(self.mapping, ids)}

    def decode(self, label: str) -> str:
        return self.mapping[label]


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    m = max(logits)
    e = [math.exp((x - m) / temperature) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def _margin(p: list[float]) -> float:
    a, b = sorted(p, reverse=True)[:2]
    return a - b


CONFIDENCE = {
    "normalized_entropy": lambda p: 1 - entropy(p) / math.log(len(p)),
    "top_probability": max,
    "top1_top2_margin": _margin,
}


def calibration_key(cfg: ModelConfig, primitive: str) -> str:
    return f"{cfg.served_model_name}/{cfg.thinking_mode or 'none'}/{primitive}/{prompts.VERSION}"


def load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except FileNotFoundError:
        return {}


async def answer_question(backend, cfg: ModelConfig, state, q, calibration: dict | None) -> tuple[dict, dict]:
    opts = prompts.options(q)
    enc = CandidateEncoder([value for value, _ in opts])
    token_ids = await enc.validate_single_token(backend)
    prompt = prompts.build(state, q, [(label, text) for label, (_, text) in zip(enc.mapping, opts)])

    t0 = time.perf_counter()
    scored = await backend.score_candidates(prompt, token_ids, cfg)
    latency_ms = (time.perf_counter() - t0) * 1000

    labels = list(enc.mapping)
    logits = [scored.logprobs[label] for label in labels]
    raw = softmax(logits)
    cal = (calibration or {}).get(calibration_key(cfg, q.type))
    p = softmax(logits, cal["temperature"]) if cal else raw
    probs = {enc.decode(label): x for label, x in zip(labels, p)}
    confidence = CONFIDENCE[cfg.confidence](p)

    if q.type == "noul":
        answer = {"type": "noul", "noul": probs["true"]}
    elif q.type == "choice":
        answer = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": confidence}
    else:
        answer = {"type": "score", "score": sum(int(k) * x for k, x in probs.items()),
                  "legend": {str(i): prompts.render(d) for i, d in enumerate(q.criteria)},
                  "probabilities": probs, "confidence": confidence}

    meta = {
        "probability_source": "token_logprobs",
        "calibrated": bool(cal),
        "calibration_method": cal["method"] if cal else None,
        "calibration_version": cal["version"] if cal else None,
        "calibration_key": calibration_key(cfg, q.type),
        "temperature": cal["temperature"] if cal else 1.0,
        "candidate_mapping": enc.mapping,
        "token_ids": token_ids,
        "raw_logprobs": dict(zip(labels, logits)),
        # Share of the model's full next-token mass that landed on valid labels; low means format trouble.
        "candidate_mass": sum(math.exp(x) for x in logits),
        "raw_probabilities": {enc.decode(label): x for label, x in zip(labels, raw)},
        "calibrated_probabilities": probs if cal else None,
        "confidence_strategy": cfg.confidence,
        "confidence": confidence,
        "entropy": entropy(p),
        "top_probability": max(p),
        "top1_top2_margin": _margin(p),
        "latency_ms": latency_ms,
        "usage": scored.usage,
        "prompt": prompt,
        "backend_debug": scored.debug,
    }
    return answer, meta


async def run(req, cfg: ModelConfig) -> dict:
    backend = backend_for(cfg)
    if not (backend.capabilities.supports_logprobs and cfg.supports_logprobs):
        raise UnsupportedQuestion(f"{cfg.id} exposes no logprobs; refusing to fall back to self-reported probabilities")
    calibration = load_calibration() if req.calibrate else None

    t0 = time.perf_counter()
    # One independent backend call per question, all in flight at once; vLLM batches them.
    results = await asyncio.gather(*(answer_question(backend, cfg, req.state, q, calibration) for q in req.questions.values()))
    total_ms = (time.perf_counter() - t0) * 1000

    metas = dict(zip(req.questions, (m for _, m in results)))
    resp = {
        "model": cfg.id,
        "answers": dict(zip(req.questions, (a for a, _ in results))),
        "usage": {k: sum(m["usage"][k] for m in metas.values()) for k in ("prompt_tokens", "completion_tokens")},
        "timing": {"total_ms": total_ms, "questions_ms": {qid: m["latency_ms"] for qid, m in metas.items()}},
        "metadata": {
            "backend": cfg.backend,
            "served_model_name": cfg.served_model_name,
            "thinking_mode": cfg.thinking_mode,
            "prompt_template_version": prompts.VERSION,
            "questions": metas,
        },
    }
    if req.trace if req.trace is not None else os.environ.get("SYSTEMONE_TRACE") == "1":
        _append_trace({"ts": time.time(), "request": req.model_dump(), "model_config": cfg.model_dump(),
                       "backend_capabilities": vars(backend.capabilities), "response": resp})
    return resp


def _append_trace(record: dict):
    # Synchronous on purpose: no await between open and write, so concurrent requests never interleave lines.
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
