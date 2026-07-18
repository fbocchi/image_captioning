import json
from pathlib import Path
from typing import Any

import numpy as np

from keras.layers import TextVectorization


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_splits(path: Path) -> dict[str, dict[str, list[str]]]:
    return _load_json(path)


def load_split(path: Path, split_name: str) -> dict[str, list[str]]:
    splits = load_splits(path)
    return splits[split_name]


def load_split_images(path: Path, split_name: str) -> list[str]:
    split = load_split(path, split_name)
    return list(split.keys())


def load_split_captions(path: Path, split_name: str) -> list[str]:
    split = load_split(path, split_name)

    captions = []

    for _, image_captions in split.items():
        captions.extend(image_captions)

    return captions


def load_vectorizer_config(path: Path) -> dict:
    return _load_json(path)


def load_vectorizer(path: Path) -> TextVectorization:
    return TextVectorization.from_config(load_vectorizer_config(path))


def load_features(path: Path) -> dict[str, np.ndarray]:
    return np.load(path, allow_pickle=True).item()


def load_history(path: Path) -> dict[str, list[float]]:
    return _load_json(path)


def load_bleu_scores(path: Path) -> dict[str, float]:
    return _load_json(path)


def load_predictions(path: Path) -> dict[str, str]:
    return _load_json(path)