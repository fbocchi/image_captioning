from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def show_predictions(
    predictions: dict,
    images_dir: Path,
    output_dir: Path | None = None,
    number_of_examples: int = 10,
) -> None:

    if output_dir is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for index, (image_id, prediction) in enumerate(
        list(predictions.items())[:number_of_examples],
        start=1,
    ):

        image_path = images_dir / f"{image_id}.jpg"

        image = Image.open(image_path)

        plt.figure(figsize=(8, 7))

        plt.imshow(image)
        plt.axis("off")

        generated = prediction["generated"]

        reference = prediction["reference"][0]

        plt.figtext(
            0.02,
            0.06,
            f"Generated:\n{generated}",
            fontsize=10,
            wrap=True,
        )

        plt.figtext(
            0.02,
            0.01,
            f"Reference:\n{reference}",
            fontsize=10,
            wrap=True,
        )

        plt.tight_layout()

        if output_dir is not None:

            plt.savefig(
                output_dir / f"prediction_{index:02d}.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

        plt.close()