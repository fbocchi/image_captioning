from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import SPLITS_FILE
from image_captioning.utils import load_split_captions


def analyze_caption_lengths(captions: list[str]) -> None:

    lengths = np.array([
        len(caption.split())
        for caption in captions
    ])

    print("=" * 60)
    print("Caption length statistics")
    print("=" * 60)

    print(f"Number of captions : {len(lengths)}")
    print(f"Minimum            : {lengths.min()}")
    print(f"Maximum            : {lengths.max()}")
    print(f"Mean               : {lengths.mean():.2f}")
    print(f"Median             : {np.median(lengths):.2f}")
    print(f"Std. deviation     : {lengths.std():.2f}")

    print("\nPercentiles")
    print("-" * 60)

    for p in [50, 75, 90, 95, 99]:
        value = np.percentile(lengths, p)
        print(f"{p:>2}% : {value:.1f}")

    print("\nCaptions longer than...")
    print("-" * 60)

    for threshold in [15, 20, 25, 30]:
        n = np.sum(lengths > threshold)
        perc = 100 * n / len(lengths)
        print(f">{threshold:2d} tokens : {n:5d} ({perc:5.2f}%)")

    plt.figure(figsize=(8, 5))

    bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5

    plt.hist(lengths, bins=bins)

    plt.xlabel("Caption length (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of caption lengths")

    plt.xticks(range(lengths.min(), lengths.max() + 1))

    plt.tight_layout()
    plt.show()


def load_training_captions(path: Path) -> list[str]:
    return load_split_captions(path, "train")


def main() -> None:
    captions = load_training_captions(SPLITS_FILE)
    analyze_caption_lengths(captions)

if __name__ == "__main__":
    main()