"""Draw the results chart used in the README (needs the optional `plots` dependency group).

  uv run --group plots python scripts/make_figures.py

The numbers come from the same Evaluation object and bootstrap code as results/report_test.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from jevbench import metrics
from jevbench.evaluate import Evaluation

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "ndcg_by_reranker.png"

NAMES = ["none", "flashrank", "cohere", "cohere_pro", "jev_pair"]
LABELS = {
    "none": "RRF order (no reranker)",
    "flashrank": "FlashRank MiniLM",
    "cohere": "Cohere Rerank 4 Fast",
    "cohere_pro": "Cohere Rerank 4 Pro",
    "jev_pair": "Jev (one request per passage)",
}
COLORS = {
    "none": "#b8bec9",
    "flashrank": "#8da4c4",
    "cohere": "#5b84b1",
    "cohere_pro": "#355f8c",
    "jev_pair": "#e8590c",
}
TYPES = ["fact", "paraphrase", "multi", "table"]


def main() -> None:
    plt.switch_backend("Agg")
    ev = Evaluation.load("test", NAMES)
    values = {name: ev.metric_values(name, ev.addressable) for name in NAMES}
    n = len(ev.addressable)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # left: overall nDCG@10 with 95% bootstrap intervals
    for i, name in enumerate(NAMES):
        mean, lo, hi = metrics.bootstrap_ci(values[name][0]["ndcg@10"])
        left.barh(i, 100 * mean, color=COLORS[name], height=0.62)
        left.errorbar(
            100 * mean,
            i,
            xerr=[[100 * (mean - lo)], [100 * (hi - mean)]],
            color="#222222",
            capsize=3,
            linewidth=1.2,
        )
        left.text(100 * hi + 1.5, i, f"{100 * mean:.1f}", va="center", fontsize=10, color="#222222")
    left.set_yticks(range(len(NAMES)), [LABELS[name] for name in NAMES])
    left.invert_yaxis()
    left.set_xlim(0, 108)
    left.set_xlabel("nDCG@10 (%)")
    left.set_title(f"Overall (n={n} questions), 95% bootstrap CI", fontsize=11, loc="left")

    # right: the same metric by question type
    width = 0.16
    for k, name in enumerate(NAMES):
        heights = []
        for t in TYPES:
            scores = values[name][1][t]["ndcg@10"]
            heights.append(100 * sum(scores) / len(scores))
        xs = [j + (k - (len(NAMES) - 1) / 2) * width for j in range(len(TYPES))]
        right.bar(xs, heights, width=width * 0.94, color=COLORS[name], label=LABELS[name])
    counts = [len(values[NAMES[0]][1][t]["ndcg@10"]) for t in TYPES]
    right.set_xticks(
        range(len(TYPES)), [f"{t}\n(n={c})" for t, c in zip(TYPES, counts, strict=True)]
    )
    right.set_ylim(0, 105)
    right.set_ylabel("nDCG@10 (%)")
    right.set_title("By question type (means; small n, no intervals)", fontsize=11, loc="left")

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is left else "y", color="#e6e8ec", linewidth=0.8)
        ax.set_axisbelow(True)

    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(NAMES), frameon=False, fontsize=9)
    fig.suptitle(
        "Reranking quality on the test split (questions whose answer is among the 30 candidates)",
        fontsize=12.5,
        x=0.01,
        y=0.985,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
