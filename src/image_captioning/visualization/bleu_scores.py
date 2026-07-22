from pathlib import Path

from matplotlib import pyplot as plt


def plot_bleu_scores(
    scores: dict[str, float],
    output_file: Path,
) -> None:

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6, 4))

    bars = plt.bar(
        labels,
        values,
        edgecolor="black",
        linewidth=1.0,
    )

    # Valore sopra ciascuna barra
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylim(0.0, 1.05)

    plt.title("BLEU")
    plt.ylabel("Score")

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()