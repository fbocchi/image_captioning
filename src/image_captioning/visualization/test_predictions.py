from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def save_predictions(
    predictions: dict[str, str],
    images_dir: Path,
    output_dir: Path,
    references: dict[str, list[str]] | None = None,
    number_of_examples: int = 10,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, (image_id, generated) in enumerate(
        list(predictions.items())[:number_of_examples],
        start=1,
    ):

        image_path = images_dir / f"{image_id}.jpg"

        image = Image.open(image_path)

        plt.figure(figsize=(8, 7))

        plt.imshow(image)
        plt.axis("off")

        plt.figtext(
            0.02,
            0.06,
            f"Generated:\n{generated}",
            fontsize=10,
            wrap=True,
        )

        if references is not None:

            plt.figtext(
                0.02,
                0.01,
                f"Reference:\n{references[image_id][0]}",
                fontsize=10,
                wrap=True,
            )

        plt.tight_layout()

        plt.savefig(
            output_dir / f"prediction_{index:02d}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()