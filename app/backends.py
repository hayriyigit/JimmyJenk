"""Inference backends. The decision core only sees DecisionBackend; nothing above this file knows about HTTP or OpenAI."""

from dataclasses import dataclass, field
from typing import Protocol

import httpx


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Capabilities:
    supports_logprobs: bool
    supports_constrained_decoding: bool
    supports_batching: bool
    supports_multimodal: bool
    supports_thinking_modes: bool


@dataclass
class Scored:
    logprobs: dict[str, float]  # label -> log p(label token | prompt), normalized over the full vocabulary
    usage: dict[str, int]
    debug: dict = field(default_factory=dict)


class DecisionBackend(Protocol):
    capabilities: Capabilities

    async def token_ids(self, text: str) -> list[int]: ...

    async def score_candidates(self, prompt: str, candidates: dict[str, int], config) -> Scored:
        """Return the model's next-token logprob for each candidate label (label -> token id)."""
        ...


class _HTTPBackend:
    name = "backend"

    def __init__(self, cfg):
        self.url = cfg.url
        self.timeout = cfg.timeout_s
        self.model = cfg.served_model_name
        self._tokens: dict[str, list[int]] = {}

    async def _post(self, path: str, body: dict) -> dict:
        # A client per call: a shared AsyncClient is bound to one event loop and breaks under repeated asyncio.run().
        try:
            async with httpx.AsyncClient(base_url=self.url, timeout=self.timeout) as http:
                r = await http.post(path, json=body)
        except httpx.HTTPError as e:
            raise BackendError(f"{self.name} at {self.url} unreachable (is its server running?): {e!r}") from e
        if r.status_code != 200:
            raise BackendError(f"{self.name} {path} returned HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    async def token_ids(self, text: str) -> list[int]:
        if text not in self._tokens:
            self._tokens[text] = await self._tokenize(text)
        return self._tokens[text]


async def reachable(url: str) -> bool:
    """True if anything answers at url; any HTTP status counts, only connection failures do not."""
    try:
        async with httpx.AsyncClient(timeout=1) as http:
            await http.get(url)
        return True
    except httpx.HTTPError:
        return False


def _messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


class VLLMBackend(_HTTPBackend):
    name = "vLLM"
    capabilities = Capabilities(
        supports_logprobs=True,
        supports_constrained_decoding=True,
        supports_batching=True,
        supports_multimodal=True,
        supports_thinking_modes=True,
    )

    async def _tokenize(self, text: str) -> list[int]:
        root = self.url.removesuffix("/v1")  # /tokenize is served next to /v1, not under it
        r = await self._post(f"{root}/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
        return r["tokens"]

    async def score_candidates(self, prompt: str, candidates: dict[str, int], config) -> Scored:
        ids = list(candidates.values())
        chat = {
            "model": self.model,
            "messages": _messages(prompt),
            "chat_template_kwargs": config.chat_template_kwargs,
            "temperature": 0,
        }
        # logprob_token_ids returns exact raw logprobs for exactly these ids, not a top-k that may miss some.
        score = {"max_tokens": 1, "logprobs": True, "logprob_token_ids": ids, "return_tokens_as_token_ids": True}
        debug = {}

        if not config.think_end:
            r = await self._post("/chat/completions", chat | score)
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            by_id = {t["token"]: t["logprob"] for t in top}
            usage = r["usage"]
        else:
            # Let the model think, then append the think-end text ourselves so the next
            # position is the answer, and read the label distribution there.
            end_ids = await self.token_ids(config.think_end)
            r1 = await self._post(
                "/chat/completions",
                chat | {"max_tokens": config.thinking_budget, "stop_token_ids": end_ids[:1], "return_token_ids": True},
            )
            c = r1["choices"][0]
            out = c["token_ids"]
            if out and out[-1] == end_ids[0]:
                out = out[:-1]
            debug = {"thinking": c["message"]["content"], "thinking_tokens": len(out),
                     "thinking_truncated": c["finish_reason"] == "length"}
            score["logprobs"] = 1  # the completions API wants an int; logprob_token_ids overrides it
            r2 = await self._post("/completions", {"model": self.model, "prompt": r1["prompt_token_ids"] + out + end_ids,
                                                   "temperature": 0} | score)
            by_id = r2["choices"][0]["logprobs"]["top_logprobs"][0]
            usage = {"prompt_tokens": r1["usage"]["prompt_tokens"],
                     "completion_tokens": r1["usage"]["completion_tokens"] + r2["usage"]["completion_tokens"]}

        try:
            logprobs = {label: by_id[f"token_id:{tid}"] for label, tid in candidates.items()}
        except KeyError as e:
            raise BackendError(f"vLLM did not return a logprob for candidate token {e}; got {sorted(by_id)}") from e
        return Scored(logprobs, {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"]}, debug)


class LlamaCppBackend(_HTTPBackend):
    """llama.cpp's llama-server (native endpoints, base_url without /v1)."""

    name = "llama.cpp"
    capabilities = Capabilities(
        supports_logprobs=True,
        supports_constrained_decoding=True,
        supports_batching=True,
        supports_multimodal=True,
        supports_thinking_modes=True,
    )
    # llama-server cannot return logprobs for chosen token ids, only the full-vocab top-N.
    # ponytail: a label outside the top-N gets the N-th logprob as an upper bound (flagged in debug);
    # raise TOP_N, or add a logit_bias pass for exact relative logits, if that flag shows up often.
    TOP_N = 100

    async def _tokenize(self, text: str) -> list[int]:
        return (await self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True}))["tokens"]

    async def score_candidates(self, prompt: str, candidates: dict[str, int], config) -> Scored:
        rendered = (await self._post("/apply-template", {"messages": _messages(prompt),
                                                         "chat_template_kwargs": config.chat_template_kwargs}))["prompt"]
        # cache_prompt off: reusing another request's KV cache made even a repeated identical question drift by
        # up to 0.02; without it repeats are bit-identical. The cache saved little, since the state sits near the top.
        score = {"n_predict": 1, "n_probs": self.TOP_N, "temperature": 0, "cache_prompt": False}
        debug = {}

        if not config.think_end:
            r = await self._post("/completion", {"prompt": rendered} | score)
            usage = {"prompt_tokens": r["tokens_evaluated"], "completion_tokens": r["tokens_predicted"]}
        else:
            # Same two phases as vLLM: think until think_end, append it as token ids, score the next position.
            end_ids = await self.token_ids(config.think_end)
            r1 = await self._post("/completion", {"prompt": rendered, "n_predict": config.thinking_budget, "temperature": 0,
                                                  "stop": [config.think_end.strip()], "return_tokens": True, "cache_prompt": False})
            out = r1["tokens"]
            if out and out[-1] == end_ids[0]:
                out = out[:-1]
            debug = {"thinking": r1["content"], "thinking_tokens": len(out), "thinking_truncated": r1["stop_type"] == "limit"}
            r = await self._post("/completion", {"prompt": [rendered, *out, *end_ids]} | score)
            usage = {"prompt_tokens": r1["tokens_evaluated"], "completion_tokens": r1["tokens_predicted"] + r["tokens_predicted"]}

        top = {t["id"]: t["logprob"] for t in r["completion_probabilities"][0]["top_logprobs"]}
        floor = min(top.values())
        missing = [label for label, tid in candidates.items() if tid not in top]
        if missing:
            debug |= {"labels_below_top_n": missing, "top_n": self.TOP_N, "logprob_is_upper_bound": True}
        return Scored({label: top.get(tid, floor) for label, tid in candidates.items()}, usage, debug)


# ponytail: a Transformers backend is still Phase 4; add a class with token_ids/score_candidates and register it here.
BACKENDS = {"vllm": VLLMBackend, "llamacpp": LlamaCppBackend}
