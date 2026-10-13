"""Static plots of synthetic indices and held-later-year tracking errors."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(directory: Path, results: dict, predictions: list[dict]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, trigger in zip(axes, ("CDD", "HDD")):
        rows = [row for row in predictions if row["trigger"] == trigger]
        for split, color, marker in (("train", "#226B9D", "o"), ("test", "#C66B21", "X")):
            selected = [row for row in rows if row["split"] == split]
            ax.scatter([row["reference_dd"] for row in selected], [row["local_dd"] for row in selected],
                       color=color, marker=marker, s=45, alpha=.85, label=split.title() + " years")
        x = np.linspace(min(row["reference_dd"] for row in rows), max(row["reference_dd"] for row in rows), 80)
        model = results[trigger]["model"]
        ax.plot(x, model["alpha"] + model["beta"] * x, color="#555555", label="Fit from train years")
        ax.set(title=trigger, xlabel="Reference index (°F-days)", ylabel="Local index (°F-days)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Synthetic seasonal indices | Train 2005–2018; test 2019–2024", fontsize=13)
    fig.savefig(Path(directory) / "index_relationship.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, trigger in zip(axes, ("CDD", "HDD")):
        rows = [row for row in predictions if row["trigger"] == trigger and row["split"] == "test"]
        years = [row["season_year"] for row in rows]
        ax.plot(years, [row["baseline_error_dd"] for row in rows], "o-", color="#226B9D", label="Local minus train mean")
        ax.plot(years, [row["tracking_error_dd"] for row in rows], "s-", color="#C66B21", label="Local minus fitted index")
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=f"{trigger} | test variance ratio = {results[trigger]['test']['residual_variance_ratio']:.3f}",
               xlabel="Season year", ylabel="Tracking error (°F-days)", xticks=years)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Synthetic test-period tracking errors | Coefficients fixed from training", fontsize=13)
    fig.savefig(Path(directory) / "test_tracking_errors.png", dpi=160)
    plt.close(fig)
