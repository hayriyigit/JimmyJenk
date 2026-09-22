# Open System-One

A local test harness for Jev-style decisions (**Noul**, **Choice** and **Score**) that can use any
generative model as the inference backend.

The model never writes out a probability. Each answer option is mapped to an opaque single-token label
(`A`, `B`, `C`, …). The harness reads the model's next-token logprob for each label and renormalizes over
the labels. What you get back is the model's own token distribution, not a self-reported confidence.

The goal is to measure whether a general LLM's logits can serve as useful System-One decisions: whether
they are accurate, whether they are calibrated, and whether they hold up when the input is paraphrased,
padded or adversarial.

> Not affiliated with TypeSafe. Jev, Noul, Choice and Score refer to the public semantics of TypeSafe's
> API. This project does not reproduce TypeSafe's proprietary confidence or calibration algorithms.

## How it works

1. **One prompt per question.** Every question is compiled separately against the same state: state,
   instructions, lettered options, and a rule to answer with exactly one letter. Question ids never
   appear in the prompt, and the questions in a request run concurrently.
2. **Opaque labels.** Options become `A`, `B`, `C`, …, and the active tokenizer checks that each label is
   exactly one token. The model can only pick among the options you supply, so it can never invent a
   Choice value.
3. **Logprobs, renormalized.** The backend returns `log p(label)` for every label, and the harness applies
   softmax over the labels.
   - Noul: `P(yes)`.
   - Choice: the argmax, plus the full distribution.
   - Score: `Σ level · P(level)`, for 2–10 levels.
4. **Thinking modes.** For R-4B's `auto` and `long` modes, the model first generates until `</think>`.
   The harness then appends `</think>\n\n` as token ids and scores the next position.
5. **Confidence and calibration.** Confidence defaults to `1 − H(p) / log K` (normalized entropy). You
   can switch it to top probability or top-1/top-2 margin per model. Temperature scaling is optional, and
   calibrations are stored per model, thinking mode, primitive and prompt-template version. Every answer
   is marked raw or calibrated, so raw logits are never presented as calibrated.

## Setup

Requirements:

- Python 3.11+.
- A model server:
  - **vLLM** with `logprob_token_ids` support (tested with 0.27.1), or
  - **llama.cpp** `llama-server` (tested with build 10639).
- Tested on a single RTX 3060 12 GB.

```sh
python -m venv .venv          # add --system-site-packages to reuse a system-wide torch/vLLM install
.venv/bin/pip install fastapi uvicorn httpx "pydantic>=2" pyyaml numpy matplotlib pytest
```

Start **one** model server. The two don't fit on a 12 GB card together.

```sh
scripts/serve_r4b.sh          # vLLM + YannQi/R-4B (bf16) on :8000
scripts/serve_r4b_gguf.sh     # llama.cpp + R-4B Q4_K_M GGUF on :8081
```

Then start the API and open the playground:

```sh
.venv/bin/python -m uvicorn app.main:app --port 8100
# open http://localhost:8100
```

Both scripts download weights into `data/cache` on first run and pass extra flags through to the server.

| Variable | Default | Effect |
|---|---|---|
| `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_UTIL`, `VLLM_PORT` | `8192`, `0.93`, `8000` | vLLM limits; the defaults fit R-4B bf16 on 12 GB |
| `R4B_GGUF_FILE` | `R-4B-Q4_K_M.gguf` | quantization to serve, e.g. `R-4B-Q8_0.gguf` |
| `LLAMA_PORT`, `LLAMA_CTX`, `LLAMA_PARALLEL`, `LLAMA_GPU_LAYERS` | `8081`, `32768`, `4`, `99` | llama-server settings; each slot gets `LLAMA_CTX / LLAMA_PARALLEL` tokens |
| `VLLM_BASE_URL`, `LLAMA_BASE_URL` | from the model config | where the API looks for each server |
| `SYSTEMONE_TRACE=1`, `SYSTEMONE_TRACE_PATH` | off, `data/traces/traces.jsonl` | append every request, prompt, label mapping, logprob and answer as JSONL |
| `SYSTEMONE_CALIBRATION` | `calibration.json` | which temperature file to load |

## API

### `POST /v1/systemone`

```sh
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "model": "r4b-q4",
  "state": {"message": "My card was charged twice and I want the extra payment back."},
  "questions": {
    "refund_requested": {"type": "noul", "instructions": "Does the customer request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "payment/refund issues", "technical": "bugs/API problems",
                                "account": "login/account issues", "other": "none"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["no urgency", "time-sensitive but not critical", "immediate/critical"]}}}'
```

| Field | Type | Notes |
|---|---|---|
| `model` | string | config id or alias; default `r4b` |
| `state` | string, object or array | the data the questions are about |
| `questions.<id>.type` | `noul`, `choice` or `score` | |
| `questions.<id>.instructions` | string, object or array | |
| `questions.<id>.criteria` | depends on type | Noul: optional `{"true": …, "false": …}`. Choice: `{option: description}` with 2–26 options. Score: a list of 2–10 levels, lowest first. |
| `calibrate` | bool | default `true`; applies a stored temperature if one exists |
| `trace` | bool | overrides `SYSTEMONE_TRACE` for this request |

Response from R-4B Q4 (real output, rounded, metadata trimmed):

```json
{
  "model": "r4b-q4-short",
  "answers": {
    "refund_requested": {"type": "noul", "noul": 0.999},
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 1.0, "technical": 0.0, "account": 0.0, "other": 0.0},
                   "confidence": 0.999},
    "urgency": {"type": "score", "score": 1.405,
                "legend": {"0": "no urgency", "1": "time-sensitive but not critical", "2": "immediate/critical"},
                "probabilities": {"0": 0.003, "1": 0.589, "2": 0.408}, "confidence": 0.367}
  },
  "usage": {"prompt_tokens": 418, "completion_tokens": 3},
  "timing": {"total_ms": 398},
  "metadata": {
    "backend": "llamacpp", "thinking_mode": "short", "prompt_template_version": "v1",
    "questions": {
      "department": {
        "probability_source": "token_logprobs", "calibrated": false, "calibration_method": null,
        "candidate_mapping": {"A": "billing", "B": "technical", "C": "account", "D": "other"},
        "raw_logprobs": {"A": -0.093, "B": -10.34, "C": -10.44, "D": -10.314},
        "candidate_mass": 0.912, "entropy": 0.001, "top_probability": 1.0, "top1_top2_margin": 1.0
      }
    }
  }
}
```

`metadata.questions.<id>` also holds:

- the label token ids and the compiled prompt;
- raw and calibrated probabilities, and the temperature applied;
- per-question latency and token usage;
- for thinking modes, the thinking text.

`candidate_mass` is the share of the model's next-token probability that landed on a valid label. A low
value means the model wanted to say something else.

Errors:

- **404:** unknown model.
- **422:** invalid criteria, or options the single-token path cannot encode.
- **502:** the model server is not running.

### `GET /v1/models`

Lists every config with its capabilities and an `online` flag showing whether its server answers.

## Playground

`http://localhost:8100` is a small single-file UI served by the API.

- **Editors:** edit the state and questions either as a form or as raw JSON. Questions JSON is the API's
  `questions` object, and may carry an optional `"expected"` per question. It is stripped before sending
  and used only for the ✓/✗ marks.
- **Comparison:** tick several models to run the same request side by side. Each column shows
  probability bars, confidence, label mass, latency and thinking tokens, plus a correct/wrong mark when
  an expected answer is set.
- **Offline models:** models whose server is down are shown as offline and can't be selected.
- **Raw tab:** the complete request and response JSON.

## Models and backends

| Config | Backend | Thinking |
|---|---|---|
| `r4b-short` (alias `r4b`), `r4b-auto`, `r4b-long` | vLLM, `YannQi/R-4B` bf16 | short / auto / long |
| `r4b-q4-short` (alias `r4b-q4`), `r4b-q4-auto`, `r4b-q4-long` | llama.cpp, `infil00p/R-4B-GGUF` Q4_K_M | short / auto / long |

The two backends read logprobs differently:

- **`VLLMBackend`** asks for exactly the label token ids (`logprob_token_ids`), so every label's logprob
  is exact.
- **`LlamaCppBackend`** reads the full-vocabulary top 100 because llama-server cannot return chosen ids.
  A label outside the top 100 gets the 100th logprob as an upper bound and is flagged as
  `labels_below_top_n`. In the benchmark below this never happened.
- Prompt caching is off on llama.cpp. With it on, identical requests were not reproducible.

The GGUF's README asks for a custom llama.cpp branch, but that branch is only needed for vision. The
GGUF declares a standard `qwen3` architecture and embeds R-4B's thinking-mode chat template, so upstream
llama.cpp serves it text-only.

**Adding a model means adding a file to `models/`:**

```yaml
id: qwen3-4b-nothink
backend: vllm                      # vllm | llamacpp
served_model_name: qwen3-4b        # also part of the calibration key
base_url_env: VLLM_BASE_URL
base_url: http://localhost:8000/v1 # llama.cpp: http://host:port without /v1
thinking_mode: none                # label used in metadata and calibration keys
chat_template_kwargs: {enable_thinking: false}
# think_end: "</think>\n\n"        # set for reasoning modes: think first, then score
# thinking_budget: 4096
# confidence: normalized_entropy   # or top_probability | top1_top2_margin
```

## Evaluation

```sh
.venv/bin/python -m eval.run eval/datasets/*.jsonl --models r4b-q4-short,r4b-q4-auto --invariance
```

The runner writes the following to `data/reports/<timestamp>/`:

- `report.md` and `report.json`, with one column per model;
- `cases.jsonl`, with one record per case;
- `reliability.png` and `risk_coverage.png`.

| Flag | Effect |
|---|---|
| `--models a,b,c` | compare several configs on the same cases |
| `--invariance` | also run the perturbation suites below |
| `--threshold 0.5` | Noul decision threshold |
| `--raw` | ignore stored calibration |
| `--fit-calibration` | fit a temperature per model and primitive on these cases and write `calibration.json` |
| `--concurrency 8` | requests in flight |

The report covers four areas:

- **Correctness:**
  - Noul: accuracy and ROC AUC.
  - Choice: accuracy and macro F1.
  - Score: MAE, level accuracy and adjacent-level accuracy.
- **Calibration**, reported separately from correctness: NLL, Brier, ECE and reliability bins.
- **Risk-coverage:** accuracy and coverage when acting only at confidence ≥ 0.5 … 0.95. This shows
  whether confidence is useful for automation.
- **Robustness and system:**
  - agreement across paraphrase groups;
  - confidence on ambiguous and missing-information cases;
  - flip rates under choice-order permutation, irrelevant-state injection and extra questions in the
    same request;
  - accuracy with 500 and 2,000 filler words added;
  - p50/p95 latency, throughput and token counts.

**Datasets.** There are 90 seed cases in three files:

- `support.jsonl`: routing, refund and urgency, with some Turkish cases;
- `jaggedness.jsonl`: the Jev failure categories (literal interpretation, math/counting, dates,
  indirection, irrelevant context, adversarial content, contradictory criteria);
- `robustness.jsonl`: missing information, ambiguity, contradictory evidence, negation, prompt
  injection, instruction and criterion paraphrases.

One JSON object per line:

```json
{"id": "sup-urg-07", "state": {"message": "..."}, "question": {"type": "score", "instructions": "...", "criteria": ["..."]},
 "expected": [0, 1], "tags": {"domain": "support", "difficulty": "medium", "language": "en", "category": "urgency"}, "group": null}
```

`expected` takes a different form per type:

- a boolean for Noul;
- an option name for Choice;
- a level, or an inclusive `[lo, hi]` range, for Score;
- `null` when there is no single right answer (ambiguous cases, where low confidence is what you want).

Cases that share a `group` are paraphrases of each other and should get the same answer.

**Calibration.** Fit on one dataset and evaluate on another:

```sh
.venv/bin/python -m eval.run fit.jsonl --models r4b-short --fit-calibration
.venv/bin/python -m eval.run held_out.jsonl --models r4b-short
```

If the fit set has no confident errors (every case correct), NLL keeps falling as the temperature goes
to 0. That primitive is skipped with a message instead of being given a degenerate temperature. No
calibration ships with the repo, so every answer is currently raw.

## Results so far

R-4B on one RTX 3060, 90 seed cases, uncalibrated. The seed set is small and mostly easy, so treat this
as a smoke test, not a benchmark.

| | bf16 short | bf16 auto | bf16 long | Q4 short |
|---|---|---|---|---|
| Noul / Choice / Score-level accuracy | 0.90 / 1.00 / 0.90 | 0.90 / 1.00 / 0.95 | 1.00 / 0.94 / 0.80 | 0.93 / 0.94 / 0.90 |
| accuracy at confidence ≥ 0.9 (coverage) | 1.00 (72%) | 0.98 (75%) | 0.93 (99%) | 0.98 (73%) |
| mean confidence on ambiguous / missing-info cases | 0.84 | 0.84 | 0.98 | 0.84 |
| flips under choice-order permutation / irrelevant state | 6% / 0% | – | – | 6% / 6% |
| p50 latency / questions per second | 280 ms / 24.5 | 220 ms / 11.2 | 4.2 s / 1.7 | 270 ms / 14.1 |

- **Short-mode confidence helps with automation; confidence after thinking doesn't.** In long mode,
  confidence collapses to about 1.0 even on wrong answers, so a confidence threshold filters almost
  nothing.
- **Auto mode rarely thinks** on these cases (about 8 tokens), so it behaves like short mode.
- **Q4 keeps the shape of bf16.** It is a little more sensitive to injected irrelevant state, and its
  thinking sometimes switches into Chinese mid-thought.
- **Extra questions in a request shift answers slightly.** They move probabilities by 0.002–0.006 on
  average and by up to 0.03 (bf16) or 0.08 (Q4) on near-ties, from batching on the server. No answer
  flipped. vLLM's `VLLM_BATCH_INVARIANT=1` would remove the shift, but it runs out of memory on 12 GB.
  A failed attempt can also leave `~/.cache/vllm/torch_compile_cache` broken; delete it if vLLM then
  crashes at startup.

## Tests

```sh
.venv/bin/python -m pytest
```

- **Unit tests** use a fake backend. They check typed answers, that probabilities sum to 1, that Score
  equals the expected value of its distribution, question independence, calibration, tracing and input
  validation.
- **Live tests** run the MVP acceptance request and the multi-question invariance check against each
  model server. They skip any model whose server is not running.
- `python -m eval.metrics` runs a self-check of the metrics.

## Project layout

| Path | Contents |
|---|---|
| `app/main.py` | FastAPI routes: `/v1/systemone`, `/v1/models`, and the playground |
| `app/engine.py` | question compilation, candidate encoding, softmax, confidence, calibration and tracing |
| `app/backends.py` | the `DecisionBackend` protocol, `VLLMBackend` and `LlamaCppBackend` |
| `app/prompts.py` | the prompt template; bump `VERSION` whenever the wording changes, because calibration and traces are keyed on it |
| `app/schemas.py` | request and response models |
| `app/registry.py` | loads `models/*.yaml` |
| `app/playground.html` | the playground UI |
| `eval/` | the runner, metrics, plots and seed datasets |
| `scripts/` | model-server launch scripts |

## Not built yet

- A Transformers backend.
- Scoring options that are several tokens long. Today a label that isn't a single token returns a
  clear 422 error.
- Domain datasets for routing, RAG relevance, answer grounding, safety and agent decisions.
- Vision input. The model supports it, but System-One questions are text-only for now.
