import asyncio
import json
import math

import pytest
from fastapi.testclient import TestClient

from app import engine, main
from app.backends import Capabilities, Scored
from app.schemas import Question

ACCEPTANCE = {
    "model": "r4b",
    "state": {"message": "My card was charged twice and I want the extra payment back."},
    "questions": {
        "refund_requested": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "payment/refund issues", "technical": "bugs/API problems",
                                    "account": "login/account issues", "other": "none"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["no urgency", "time-sensitive but not critical", "immediate/critical"]},
    },
}


class FakeBackend:
    capabilities = Capabilities(True, False, True, False, False)

    def __init__(self):
        self.prompts = []

    async def token_ids(self, text):
        return [ord(c) for c in text]  # one "token" per character: single letters pass, "AA" fails

    async def score_candidates(self, prompt, candidates, config):
        self.prompts.append(prompt)
        # Unnormalized, like real full-vocab logprobs over a subset.
        return Scored({label: -1.0 - 0.7 * i for i, label in enumerate(candidates)}, {"prompt_tokens": 100, "completion_tokens": 1})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    backend = FakeBackend()
    monkeypatch.setattr(engine, "backend_for", lambda cfg: backend)
    monkeypatch.setattr(engine, "CALIBRATION_PATH", tmp_path / "calibration.json")
    return backend


client = TestClient(main.app)


def test_acceptance_request_returns_typed_distributions(fake):
    r = client.post("/v1/systemone", json=ACCEPTANCE)
    assert r.status_code == 200, r.text
    body = r.json()
    a = body["answers"]

    assert a["refund_requested"]["type"] == "noul" and 0 < a["refund_requested"]["noul"] < 1
    assert "confidence" not in a["refund_requested"]

    dept = a["department"]
    assert set(dept["probabilities"]) == set(ACCEPTANCE["questions"]["department"]["criteria"])
    assert dept["choice"] in dept["probabilities"]
    assert math.isclose(sum(dept["probabilities"].values()), 1, abs_tol=1e-9)
    assert 0 <= dept["confidence"] <= 1

    urg = a["urgency"]
    assert math.isclose(sum(urg["probabilities"].values()), 1, abs_tol=1e-9)
    assert math.isclose(urg["score"], sum(int(k) * p for k, p in urg["probabilities"].items()))
    assert urg["legend"] == {"0": "no urgency", "1": "time-sensitive but not critical", "2": "immediate/critical"}

    meta = body["metadata"]["questions"]["department"]
    assert meta["candidate_mapping"] == {"A": "billing", "B": "technical", "C": "account", "D": "other"}
    assert set(meta["raw_logprobs"]) == {"A", "B", "C", "D"}
    assert meta["probability_source"] == "token_logprobs" and meta["calibrated"] is False
    assert body["usage"] == {"prompt_tokens": 300, "completion_tokens": 3}


def test_questions_are_compiled_independently(fake):
    questions = {f"qid_{i}_zx": q for i, q in enumerate(ACCEPTANCE["questions"].values())}
    client.post("/v1/systemone", json=ACCEPTANCE | {"questions": questions})
    together = sorted(fake.prompts)
    fake.prompts.clear()
    for qid, q in questions.items():
        client.post("/v1/systemone", json=ACCEPTANCE | {"questions": {qid: q}})
    assert sorted(fake.prompts) == together
    assert not any("_zx" in p for p in together)


def test_calibration_is_applied_and_reported(fake):
    engine.CALIBRATION_PATH.write_text(json.dumps({"r4b/short/choice/v1": {
        "method": "temperature_scaling", "temperature": 2.0, "version": "test"}}))
    cal = client.post("/v1/systemone", json=ACCEPTANCE).json()
    raw = client.post("/v1/systemone", json=ACCEPTANCE | {"calibrate": False}).json()

    meta = cal["metadata"]["questions"]["department"]
    assert meta["calibrated"] is True and meta["calibration_version"] == "test"
    assert cal["metadata"]["questions"]["urgency"]["calibrated"] is False  # no entry for score
    assert raw["metadata"]["questions"]["department"]["calibrated"] is False
    # T > 1 flattens: the top option loses probability.
    assert cal["answers"]["department"]["probabilities"]["billing"] < raw["answers"]["department"]["probabilities"]["billing"]
    assert meta["raw_probabilities"] == raw["answers"]["department"]["probabilities"]


def test_trace_records_prompts_and_logprobs(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "TRACE_PATH", tmp_path / "t.jsonl")
    client.post("/v1/systemone", json=ACCEPTANCE | {"trace": True})
    client.post("/v1/systemone", json=ACCEPTANCE | {"trace": False})
    (record,) = [json.loads(line) for line in engine.TRACE_PATH.read_text().splitlines()]
    q = record["response"]["metadata"]["questions"]["department"]
    assert q["prompt"] and q["raw_logprobs"] and q["candidate_mapping"] and record["model_config"]["id"] == "r4b-short"


@pytest.mark.parametrize("question", [
    {"type": "score", "instructions": "x", "criteria": ["only one"]},
    {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
    {"type": "choice", "instructions": "x", "criteria": ["a", "b"]},
    {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},
])
def test_invalid_criteria_rejected(fake, question):
    assert client.post("/v1/systemone", json={"state": "s", "questions": {"q": question}}).status_code == 422


def test_too_many_choice_options_is_a_clear_error(fake):
    q = {"type": "choice", "instructions": "x", "criteria": {f"opt{i}": "" for i in range(27)}}
    r = client.post("/v1/systemone", json={"state": "s", "questions": {"q": q}})
    assert r.status_code == 422 and "single-token" in r.text


def test_multi_token_labels_rejected():
    enc = engine.CandidateEncoder(["x", "y"])
    enc.mapping = {"AA": "x", "B": "y"}
    with pytest.raises(engine.UnsupportedQuestion):
        asyncio.run(enc.validate_single_token(FakeBackend()))


def test_confidence_strategies():
    uniform, onehot = [0.25] * 4, [1.0, 0, 0, 0]
    ne = engine.CONFIDENCE["normalized_entropy"]
    assert math.isclose(ne(uniform), 0, abs_tol=1e-12) and math.isclose(ne(onehot), 1)
    assert engine.CONFIDENCE["top1_top2_margin"]([0.6, 0.3, 0.1]) == pytest.approx(0.3)


def test_softmax_temperature_matches_definition():
    lp = [-0.1, -2.3, -5.0]
    p = engine.softmax(lp, 1.7)
    z = sum(math.exp(x / 1.7) for x in lp)
    assert p == pytest.approx([math.exp(x / 1.7) / z for x in lp])


def test_prompt_shape():
    from app import prompts
    q = Question(type="score", instructions="How urgent?", criteria=["low", "mid", "high"])
    text = prompts.build({"a": 1}, q, [("A", "low"), ("B", "mid"), ("C", "high")])
    assert "lowest (A) to highest (C)" in text and "(A, B or C)" in text and '"a": 1' in text
