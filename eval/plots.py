"""Reliability diagrams and risk-coverage curves, one line per model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def plot(report: dict, out: Path):
    models = report["models"]
    kinds = [k for k in ("noul", "choice", "score") if any(k in s["calibration"] for s in models.values())]
    if not kinds:
        return

    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 4.2), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
        for color, (mid, s) in zip(SERIES, models.items()):
            bins = s["calibration"].get(kind, {}).get("reliability", [])
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=color, linewidth=2,
                    marker="o", markersize=6, markeredgecolor="white", label=mid)
        _axes(ax, f"{kind}: reliability", "mean top-class probability", "accuracy")
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for color, (mid, s) in zip(SERIES, models.items()):
        pts = [p for p in s["risk_coverage"]["all"] if p["n"]]
        ax.plot([p["coverage"] for p in pts], [p["accuracy"] for p in pts], color=color, linewidth=2,
                marker="o", markersize=6, markeredgecolor="white", label=mid)
        if pts:  # label only the strictest threshold, where the curve ends
            ax.annotate(f"≥{pts[-1]['threshold']}", (pts[-1]["coverage"], pts[-1]["accuracy"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
    _axes(ax, "Risk-coverage (all primitives)", "coverage: share of cases acted on", "accuracy on those cases")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=130)
    plt.close(fig)
