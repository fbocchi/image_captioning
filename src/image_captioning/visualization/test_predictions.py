from pathlib import Path
import textwrap

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

        with Image.open(image_path) as image:

            fig, ax = plt.subplots(figsize=(8, 6))

            #
            # Image
            #
            ax.imshow(image)
            ax.axis("off")

            #
            # Leave just enough room for the captions
            #
            fig.subplots_adjust(bottom=0.24)

            #
            # Generated caption
            #
            generated = textwrap.fill(
                generated,
                width=75,
            )

            fig.text(
                0.02,
                0.16,
                "Generated:",
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="top",
            )

            fig.text(
                0.15,
                0.16,
                generated,
                fontsize=11,
                ha="left",
                va="top",
            )

            #
            # Reference caption
            #
            if references is not None:

                reference = textwrap.fill(
                    references[image_id][0],
                    width=75,
                )

                fig.text(
                    0.02,
                    0.08,
                    "Reference:",
                    fontsize=11,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

                fig.text(
                    0.15,
                    0.08,
                    reference,
                    fontsize=11,
                    ha="left",
                    va="top",
                )

            plt.savefig(
                output_dir / f"prediction_{index:02d}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)