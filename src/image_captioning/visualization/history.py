from pathlib import Path

from matplotlib import pyplot as plt


def plot_history(
    history: dict[str, list[float]],
    output_file: Path | None = None,
) -> None:

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["loss"],
        label="Training",
    )

    plt.plot(
        history["val_loss"],
        label="Validation",
    )

    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if output_file is not None:
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    plt.close()