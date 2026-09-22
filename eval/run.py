"""Benchmark runner.

    python -m eval.run eval/datasets/*.jsonl --models r4b-short,r4b-auto [--invariance] [--fit-calibration] [--raw]

Writes cases.jsonl, report.json, report.md and plots to --out (default data/reports/<timestamp>).
"""

import argparse
import asyncio
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app import engine
from app.registry import get_model
from app.schemas import SystemOneRequest
from eval import metrics as M

IRRELEVANT = "Unrelated note: the third-floor printer was serviced on Tuesday and the cafeteria menu changes next week."
FILLER = (
    "Log entry: routine backup completed without errors. "
    "The quarterly newsletter mentions a new plant in the lobby. "
    "Parking lot B will be repainted over the weekend. "
    "A reminder that the fire drill is scheduled for the first Monday of the month. "
    "The IT wiki page about keyboard shortcuts was updated. "
)
PAD_WORDS = (0, 500, 2000)
EXTRA_QUESTIONS = {
    "extra_noul": {"type": "noul", "instructions": "Is the state longer than one sentence?"},
    "extra_score": {"type": "score", "instructions": "How formal is the language in the state?",
                    "criteria": ["informal", "neutral", "formal"]},
}
TAG_KEYS = ("dataset", "category", "domain", "difficulty", "language")


def load(paths: list[str]) -> list[dict]:
    cases = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                case = json.loads(line)
                case["tags"] = {"dataset": Path(p).stem} | case.get("tags", {})
                cases.append(case)
    return cases


def expected_options(case) -> list[str] | None:
    e, kind = case.get("expected"), case["question"]["type"]
    if e is None:
        return None
    if kind == "noul":
        return ["true" if e else "false"]
    if kind == "score":
        lo, hi = (e, e) if isinstance(e, int) else e
        return [str(i) for i in range(lo, hi + 1)]
    return [e]


def with_state(state, extra: str):
    if isinstance(state, dict):
        return state | {"note": extra}
    if isinstance(state, list):
        return state + [extra]
    return f"{state}\n\n{extra}"


async def ask(cfg, case, args, sem, state=None, question=None, extra=None) -> dict:
    question = question or case["question"]
    req = SystemOneRequest(model=cfg.id, state=case["state"] if state is None else state,
                           questions={"q": question} | (extra or {}), calibrate=not args.raw, trace=False)
    async with sem:
        resp = await engine.run(req, cfg)
    m = resp["metadata"]["questions"]["q"]
    probs = m["calibrated_probabilities"] or m["raw_probabilities"]
    if question["type"] == "noul":
        predicted = "true" if probs["true"] >= args.threshold else "false"
    else:
        predicted = max(probs, key=probs.get)
    exp = expected_options(case)
    return {
        "id": case["id"], "model": cfg.id, "type": question["type"], "tags": case["tags"], "group": case.get("group"),
        "expected": case.get("expected"), "expected_options": exp, "probabilities": probs,
        "raw_logits": list(m["raw_logprobs"].values()), "options": list(m["candidate_mapping"].values()),
        "predicted": predicted, "argmax": max(probs, key=probs.get),
        "correct": predicted in exp if exp else None, "p_expected": sum(probs[o] for o in exp) if exp else None,
        "confidence": m["confidence"], "top_probability": m["top_probability"],
        "score": resp["answers"]["q"].get("score"), "calibrated": m["calibrated"], "candidate_mass": m["candidate_mass"],
        "latency_ms": m["latency_ms"], **m["usage"], "thinking_tokens": m["backend_debug"].get("thinking_tokens"),
        "labels_below_top_n": m["backend_debug"].get("labels_below_top_n"),
    }


def _max_dp(a: dict, b: dict) -> float:
    return max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])


async def invariance(cfg, cases, base: dict, args, sem) -> dict:
    """Perturb each case and measure how much the answer moves. Ideal: nothing moves, except accuracy under padding stays flat."""
    jobs = {}
    for c in cases:
        jobs[("irrelevant", c["id"])] = ask(cfg, c, args, sem, state=with_state(c["state"], IRRELEVANT))
        jobs[("multi", c["id"])] = ask(cfg, c, args, sem, extra=EXTRA_QUESTIONS)
        if c["question"]["type"] == "choice":
            q = c["question"] | {"criteria": dict(reversed(list(c["question"]["criteria"].items())))}
            jobs[("permutation", c["id"])] = ask(cfg, c, args, sem, question=q)
        for words in PAD_WORDS[1:]:
            pad = " ".join((FILLER * (words // 50 + 1)).split()[:words])
            jobs[(f"pad{words}", c["id"])] = ask(cfg, c, args, sem, state=with_state(c["state"], pad))
    results = dict(zip(jobs, await asyncio.gather(*jobs.values())))

    def compare(kind):
        pairs = [(base[cid], r) for (k, cid), r in results.items() if k == kind]
        return {"n": len(pairs), "flip_rate": float(np.mean([a["argmax"] != b["argmax"] for a, b in pairs])) if pairs else None,
                "mean_max_abs_dp": float(np.mean([_max_dp(a, b) for a, b in pairs])) if pairs else None,
                "max_abs_dp": max((_max_dp(a, b) for a, b in pairs), default=None)}

    long_state = {}
    for words in PAD_WORDS:
        recs = base.values() if words == 0 else [r for (k, _), r in results.items() if k == f"pad{words}"]
        long_state[str(words)] = M.accuracy([r["correct"] for r in recs if r["correct"] is not None])
    return {"choice_permutation": compare("permutation"), "irrelevant_injection": compare("irrelevant"),
            "multi_question": compare("multi"), "long_state_accuracy_by_padding_words": long_state}


def summarize(recs: list[dict], wall_s: float) -> dict:
    labeled = [r for r in recs if r["correct"] is not None]
    correctness, calibration, rc = {}, {}, {}
    for kind in ("noul", "choice", "score"):
        rs = [r for r in labeled if r["type"] == kind]
        if not rs:
            continue
        correct = [r["correct"] for r in rs]
        top_correct = [r["argmax"] in r["expected_options"] for r in rs]
        cal = {"nll": M.nll([r["p_expected"] for r in rs]), "ece": M.ece([r["top_probability"] for r in rs], top_correct),
               "reliability": M.reliability([r["top_probability"] for r in rs], top_correct)}
        if kind == "noul":
            y = [r["expected"] is True for r in rs]
            correctness[kind] = {"n": len(rs), "accuracy": M.accuracy(correct),
                                 "roc_auc": M.roc_auc([r["probabilities"]["true"] for r in rs], y)}
            cal["brier"] = float(np.mean([(r["probabilities"]["true"] - t) ** 2 for r, t in zip(rs, y)]))
        elif kind == "choice":
            correctness[kind] = {"n": len(rs), "accuracy": M.accuracy(correct),
                                 "macro_f1": M.macro_f1([r["predicted"] for r in rs], [r["expected"] for r in rs])}
            cal["brier"] = M.brier([list(r["probabilities"].values()) for r in rs],
                                   [r["options"].index(r["expected"]) for r in rs])
        else:
            dist = [min(abs(int(r["predicted"]) - int(o)) for o in r["expected_options"]) for r in rs]
            correctness[kind] = {"n": len(rs), "mae": float(np.mean([min(abs(r["score"] - int(o)) for o in r["expected_options"])
                                                                     for r in rs])),
                                 "level_accuracy": M.accuracy(correct), "adjacent_accuracy": M.accuracy([d <= 1 for d in dist])}
        calibration[kind] = cal
        rc[kind] = M.risk_coverage([r["confidence"] for r in rs], correct)
    rc["all"] = M.risk_coverage([r["confidence"] for r in labeled], [r["correct"] for r in labeled])

    by_tag = defaultdict(lambda: defaultdict(list))
    for r in labeled:
        for key in TAG_KEYS:
            if key in r["tags"]:
                by_tag[key][r["tags"][key]].append(r["correct"])
    groups = defaultdict(list)
    for r in recs:
        if r["group"]:
            groups[r["group"]].append(r)
    groups = {g: rs for g, rs in groups.items() if len(rs) > 1}
    unlabeled = [r for r in recs if r["correct"] is None]
    lat = [r["latency_ms"] for r in recs]
    return {
        "n": len(recs), "n_labeled": len(labeled), "calibrated": any(r["calibrated"] for r in recs),
        "correctness": correctness, "calibration": calibration, "risk_coverage": rc,
        "by_tag": {k: {v: {"n": len(c), "accuracy": M.accuracy(c)} for v, c in vals.items()} for k, vals in by_tag.items()},
        # Paraphrase groups: every member should get the same answer with similar probability.
        "consistency": {"groups": len(groups),
                        "agreement": M.accuracy([len({r["argmax"] for r in rs}) == 1 for rs in groups.values()]),
                        "mean_p_expected_spread": float(np.mean([max(r["p_expected"] for r in rs) - min(r["p_expected"] for r in rs)
                                                                 for rs in groups.values() if rs[0]["p_expected"] is not None] or [np.nan]))},
        # Cases with no single right answer (ambiguous / missing info): confidence here should be low.
        "unlabeled": {"n": len(unlabeled), "mean_confidence": float(np.mean([r["confidence"] for r in unlabeled])) if unlabeled else None},
        "mean_candidate_mass": float(np.mean([r["candidate_mass"] for r in recs])),
        # Backends that only expose a top-N (llama.cpp) give an upper bound, not an exact logprob, for these.
        "cases_with_labels_below_top_n": sum(bool(r["labels_below_top_n"]) for r in recs),
        "system": {"p50_latency_ms": float(np.percentile(lat, 50)), "p95_latency_ms": float(np.percentile(lat, 95)),
                   "questions_per_s": len(recs) / wall_s, "requests_per_s": len(recs) / wall_s,
                   "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
                   "completion_tokens": sum(r["completion_tokens"] for r in recs)},
    }


def fit_calibration(recs_by_model: dict, datasets: list[str], min_n: int = 10) -> dict:
    stored = engine.load_calibration()
    fitted = {}
    for model_id, recs in recs_by_model.items():
        cfg = get_model(model_id)
        for kind in ("noul", "choice", "score"):
            rs = [r for r in recs if r["type"] == kind and r["expected_options"]]
            if len(rs) < min_n:
                continue
            t, before, after = M.fit_temperature([r["raw_logits"] for r in rs],
                                                 [[r["options"].index(o) for o in r["expected_options"]] for r in rs])
            key = engine.calibration_key(cfg, kind)
            if t is None:
                print(f"skip {key}: temperature not identifiable on {len(rs)} cases (no confident errors); add harder cases")
                continue
            fitted[key] = {"method": "temperature_scaling", "temperature": t, "n": len(rs), "nll_before": before, "nll_after": after,
                           "version": f"{datetime.datetime.now(datetime.UTC).date()}:{'+'.join(Path(d).stem for d in datasets)}:n={len(rs)}"}
    engine.CALIBRATION_PATH.write_text(json.dumps(stored | fitted, indent=2) + "\n")
    return fitted


def fmt(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}" if isinstance(x, float) else str(x)


def markdown(report: dict) -> str:
    models = report["models"]
    ids = list(models)
    probs = "raw (uncalibrated)" if report["raw"] else "calibrated where a calibration exists"
    out = [f"# System-One eval — {report['created']}", "",
           f"Datasets: {', '.join(report['datasets'])}. Noul threshold {report['threshold']}. Probabilities: {probs}.", ""]

    def table(title, rows):
        out.extend([f"## {title}", "", "| metric | " + " | ".join(ids) + " |", "|---" * (len(ids) + 1) + "|"])
        out.extend(f"| {name} | " + " | ".join(fmt(get(models[m])) for m in ids) + " |" for name, get in rows)
        out.append("")

    def dig(*path):
        def get(s):
            for p in path:
                s = s.get(p) if isinstance(s, dict) else None
            return s
        return get

    corr_rows, cal_rows = [], []
    for kind, keys in {"noul": ("n", "accuracy", "roc_auc"), "choice": ("n", "accuracy", "macro_f1"),
                       "score": ("n", "mae", "level_accuracy", "adjacent_accuracy")}.items():
        corr_rows += [(f"{kind} {k}", dig("correctness", kind, k)) for k in keys]
        cal_rows += [(f"{kind} {k}", dig("calibration", kind, k)) for k in ("nll", "brier", "ece")]
    table("Correctness", corr_rows)
    table("Calibration (lower is better)", cal_rows + [("any calibrated", dig("calibrated"))])
    table("Risk-coverage: accuracy / coverage when acting only at confidence ≥ t", [
        (f"≥ {rc['threshold']}", lambda s, i=i: None if not s["risk_coverage"]["all"][i]["n"] else
         f"{fmt(s['risk_coverage']['all'][i]['accuracy'])} / {s['risk_coverage']['all'][i]['coverage']:.0%}")
        for i, rc in enumerate(models[ids[0]]["risk_coverage"]["all"])])
    cats = sorted({c for s in models.values() for c in s["by_tag"].get("category", {})})
    table("Accuracy by category", [(c, dig("by_tag", "category", c, "accuracy")) for c in cats])
    invariance_rows = [(f"{k} {m}", dig("invariance", k, m)) for k in ("choice_permutation", "irrelevant_injection", "multi_question")
                       for m in ("flip_rate", "mean_max_abs_dp")] \
        + [(f"accuracy, +{w} filler words", dig("invariance", "long_state_accuracy_by_padding_words", str(w))) for w in PAD_WORDS]
    table("Robustness", [("paraphrase agreement", dig("consistency", "agreement")),
                         ("paraphrase p(expected) spread", dig("consistency", "mean_p_expected_spread")),
                         ("ambiguous/missing: mean confidence", dig("unlabeled", "mean_confidence")),
                         ("mean candidate mass", dig("mean_candidate_mass")),
                         ("cases with a label outside top-N", dig("cases_with_labels_below_top_n"))]
          + (invariance_rows if any("invariance" in s for s in models.values()) else []))
    table("System", [(k, dig("system", k)) for k in ("p50_latency_ms", "p95_latency_ms", "questions_per_s",
                                                     "requests_per_s", "prompt_tokens", "completion_tokens")])
    return "\n".join(out)


async def main(args):
    cases = load(args.datasets)
    now = datetime.datetime.now(datetime.UTC).astimezone()
    out = Path(args.out or f"data/reports/{now:%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    report = {"created": f"{now:%Y-%m-%d %H:%M %Z}", "datasets": args.datasets, "threshold": args.threshold,
              "raw": args.raw, "models": {}}
    recs_by_model = {}
    with (out / "cases.jsonl").open("w") as f:
        for model in args.models.split(","):
            cfg = get_model(model)
            t0 = time.perf_counter()
            recs = await asyncio.gather(*(ask(cfg, c, args, sem) for c in cases))
            summary = summarize(recs, time.perf_counter() - t0)
            if args.invariance:
                summary["invariance"] = await invariance(cfg, cases, {r["id"]: r for r in recs}, args, sem)
            report["models"][cfg.id] = summary
            recs_by_model[cfg.id] = recs
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
            print(f"{cfg.id}: {len(recs)} cases in {time.perf_counter() - t0:.1f}s")
    if args.fit_calibration:
        report["fitted_calibration"] = fit_calibration(recs_by_model, args.datasets)
        print(f"wrote {len(report['fitted_calibration'])} temperatures to {engine.CALIBRATION_PATH}; "
              "evaluate on a different dataset before trusting them")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(markdown(report))
    from eval.plots import plot
    plot(report, out)
    print(markdown(report))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--models", default="r4b-short")
    ap.add_argument("--out")
    ap.add_argument("--threshold", type=float, default=0.5, help="noul decision threshold")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="ignore stored calibration")
    ap.add_argument("--invariance", action="store_true", help="also run perturbation suites")
    ap.add_argument("--fit-calibration", action="store_true", help="fit temperatures on these cases (implies --raw)")
    args = ap.parse_args()
    args.raw = args.raw or args.fit_calibration
    asyncio.run(main(args))
