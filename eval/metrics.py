"""Correctness, calibration and risk-coverage metrics. Pure numpy so they are easy to test."""

import numpy as np

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def accuracy(correct) -> float:
    return float(np.mean(correct)) if len(correct) else float("nan")


def nll(p_expected) -> float:
    return float(np.mean(-np.log(np.clip(p_expected, 1e-12, 1))))


def brier(probs: list[list[float]], expected_idx: list[int]) -> float:
    """Multiclass Brier: sum over classes of (p - onehot)^2, averaged over cases."""
    return float(np.mean([sum((q - (i == e)) ** 2 for i, q in enumerate(p)) for p, e in zip(probs, expected_idx)]))


def reliability(conf, correct, bins: int = 10) -> list[dict]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return [{"lo": b / bins, "hi": (b + 1) / bins, "n": int((idx == b).sum()),
             "confidence": float(conf[idx == b].mean()), "accuracy": float(correct[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def ece(conf, correct, bins: int = 10) -> float:
    return sum(r["n"] * abs(r["accuracy"] - r["confidence"]) for r in reliability(conf, correct, bins)) / max(len(conf), 1)


def roc_auc(scores, labels) -> float:
    # ponytail: O(pos*neg) pairwise comparison; switch to a rank formula past ~10k cases.
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def macro_f1(pred: list[str], gold: list[str]) -> float:
    f1 = []
    for label in set(gold) | set(pred):
        tp = sum(p == g == label for p, g in zip(pred, gold))
        fp = sum(p == label != g for p, g in zip(pred, gold))
        fn = sum(g == label != p for p, g in zip(pred, gold))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)) if f1 else float("nan")


def risk_coverage(conf, correct, thresholds=THRESHOLDS) -> list[dict]:
    """Accuracy when only acting on cases with confidence >= t, and how many cases that leaves."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return [{"threshold": t, "coverage": float((conf >= t).mean()) if len(conf) else float("nan"),
             "n": int((conf >= t).sum()), "accuracy": float(correct[conf >= t].mean()) if (conf >= t).any() else None}
            for t in thresholds]


def fit_temperature(logits: list[list[float]], expected_idx: list[list[int]]) -> tuple[float | None, float, float]:
    """Temperature minimising NLL of the expected option(s). Returns (T, nll at T=1, nll at T).

    T is None when the optimum sits on the search boundary: with no confident errors in the data
    (e.g. every case correct) NLL keeps falling as T -> 0, so no finite temperature is supported."""
    def mean_nll(t):
        total = 0.0
        for lg, exp in zip(logits, expected_idx):
            z = np.asarray(lg) / t
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            total -= np.log(max(p[exp].sum(), 1e-12))
        return total / len(logits)

    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    best = min(grid, key=mean_nll)
    return (None if best in (grid[0], grid[-1]) else float(best)), mean_nll(1.0), mean_nll(best)


if __name__ == "__main__":
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert ece([0.8] * 10, [1] * 8 + [0] * 2) < 1e-9
    assert brier([[1, 0], [0.5, 0.5]], [0, 0]) == 0.25
    rc = risk_coverage([0.95, 0.55, 0.3], [1, 0, 0])
    assert rc[0] == {"threshold": 0.5, "coverage": 2 / 3, "n": 2, "accuracy": 0.5} and rc[-1]["accuracy"] == 1.0
    # Overconfident logits (correct only 75% of the time at ~99% confidence) should be cooled: T > 1.
    t, before, after = fit_temperature([[5, 0]] * 3 + [[0, 5]], [[0]] * 4)
    assert t > 1 and after < before
    assert fit_temperature([[5, 0], [0, 5]], [[0], [1]])[0] is None  # all correct: not identifiable
    print("metrics ok")
