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

    plt.figure(figsize=(6, 4))

    plt.bar(
        scores.keys(),
        scores.values(),
    )

    plt.ylim(0, 1)

    plt.title("BLEU Scores")
    plt.ylabel("Score")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()