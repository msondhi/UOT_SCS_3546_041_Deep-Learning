"""Plot robustness results: accuracy vs. damage strength, one panel per degradation.

Reads the CSVs written by ``deepfake_kyc.evaluate --degradation all`` and draws one
line per model. Example:

    .venv/bin/python scripts/plot_robustness.py \
        --run "Standard training (E3)=outputs/03_effnet_finetune/eval_val_robustness.csv" \
        --run "Robust training (E6)=outputs/06_effnet_robust/eval_val_robustness.csv" \
        --output outputs/figures/robustness_val.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Panel order and plain-language titles for each degradation.
PANELS = {
    "blur": "Blur",
    "downscale": "Low resolution",
    "jpeg": "JPEG compression",
    "noise": "Noise",
    "screen": "Photo of a screen",
}

# Colours and markers: series colour in fixed order, plus a distinct marker per
# series so the lines can be told apart without colour.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_run(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Return {degradation: [(severity, accuracy), ...]}, each starting with the clean result at severity 0."""
    rows = list(csv.DictReader(open(path)))
    clean = next(float(r["accuracy"]) for r in rows if r["condition"] == "clean")
    out: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r["condition"] != "clean":
            out.setdefault(r["condition"], [(0, clean)]).append((int(r["severity"]), float(r["accuracy"])))
    return {k: sorted(v) for k, v in out.items()}


def plot(runs: dict[str, dict], output: Path, title: str) -> None:
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
    panels = list(PANELS.items()) + [("_mean", "Average of all five")]

    for ax, (key, panel_title) in zip(axes.flat, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.axhline(0.5, color=INK_2, linewidth=1)
        ax.text(-0.1, 0.485, "Guessing (50%)", va="top", ha="left", fontsize=8, color=INK_2)
        for i, (name, data) in enumerate(runs.items()):
            if key == "_mean":
                sev = [s for s, _ in data["blur"]]
                acc = [sum(data[d][j][1] for d in PANELS) / len(PANELS) for j in range(len(sev))]
            else:
                sev, acc = zip(*data[key], strict=True)
            ax.plot(sev, acc, color=SERIES_COLOURS[i], marker=SERIES_MARKERS[i], markersize=6,
                    linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
            ax.text(5.15, acc[-1], f"{acc[-1]:.0%}", va="center", ha="left", fontsize=9, color=INK)
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlim(-0.2, 5.9)
        ax.set_xticks(range(6), ["Clean", "1", "2", "3", "4", "5"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    for ax in axes[1]:
        ax.set_xlabel("Damage strength")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=len(labels), frameon=False)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=SURFACE)
    print(f"Saved {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=CSV", help="Model label and CSV path")
    parser.add_argument("--output", default="outputs/figures/robustness.png")
    parser.add_argument("--title", default="Accuracy on damaged photos: standard vs. robust training")
    args = parser.parse_args(argv)
    runs = {}
    for item in args.run:
        label, path = item.split("=", 1)
        runs[label] = read_run(path)
    plot(runs, Path(args.output), args.title)


if __name__ == "__main__":
    main()
