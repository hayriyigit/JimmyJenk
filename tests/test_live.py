"""Runs against real model servers (scripts/serve_r4b.sh, scripts/serve_r4b_gguf.sh). Each model is skipped when its server is down."""

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_engine import ACCEPTANCE

client = TestClient(app)


@pytest.fixture(params=["r4b-short", "r4b-q4-short", "r4b-q4-auto"])
def model(request):
    r = client.post("/v1/systemone", json={"model": request.param, "state": "x", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    if r.status_code == 502:
        pytest.skip(f"{request.param}: server not reachable")
    return request.param


def test_mvp_acceptance(model):
    body = client.post("/v1/systemone", json=ACCEPTANCE | {"model": model, "calibrate": False}).json()
    a, meta = body["answers"], body["metadata"]["questions"]

    assert a["refund_requested"]["noul"] > 0.5
    assert a["department"]["choice"] == "billing"
    for q in ("department", "urgency"):
        assert math.isclose(sum(a[q]["probabilities"].values()), 1, abs_tol=1e-6)
    assert math.isclose(a["urgency"]["score"], sum(int(k) * p for k, p in a["urgency"]["probabilities"].items()))
    for m in meta.values():
        assert m["raw_logprobs"] and m["candidate_mapping"] and m["candidate_mass"] > 0.5
        assert "labels_below_top_n" not in m["backend_debug"]


def test_adding_a_question_does_not_change_existing_answers(model):
    alone = client.post("/v1/systemone", json=ACCEPTANCE | {"model": model, "questions": {"d": ACCEPTANCE["questions"]["department"]}}).json()
    together = client.post("/v1/systemone", json=ACCEPTANCE | {"model": model}).json()
    p1, p2 = alone["answers"]["d"]["probabilities"], together["answers"]["department"]["probabilities"]
    # Batching can shift logits slightly; anything beyond that means the questions leak into each other.
    assert max(abs(p1[k] - p2[k]) for k in p1) < 0.02
