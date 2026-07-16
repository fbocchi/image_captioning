import json
from pathlib import Path

import numpy as np

from keras.callbacks import History
from keras.layers import TextVectorization

from image_captioning.config.paths import (
    SPLIT_FILE, TEXT_VECTORIZATION_CONFIG_FILE, FEATURES_FILE,
    HISTORY_FILE, BLEU_SCORES_FILE, TEST_PREDICTIONS_FILE
)

def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_split_json() -> dict[str, dict[str, list[str]]]:
    return _load_json(SPLIT_FILE)

def load_training_captions() -> list[str]:
    split_json = load_split_json()
    train_split = split_json["train"]
    training_captions = []
    for _, image_captions in train_split.items():
        training_captions.extend(image_captions)
    return training_captions

def load_test_split() -> dict[str, list[str]]:
    split_json = load_split_json()
    return split_json["test"]

def load_text_vectorization_config():
    return _load_json(TEXT_VECTORIZATION_CONFIG_FILE)

def load_text_vectorization() -> TextVectorization:
    return TextVectorization.from_config(
        load_text_vectorization_config()
    )

def load_vocab_size() -> int:
    return load_text_vectorization_config()["vocabulary_size"]

def load_output_sequence_length() -> int:
    return load_text_vectorization_config()["output_sequence_length"]

def load_features() -> dict[str, np.ndarray]:
    return np.load(FEATURES_FILE, allow_pickle=True).item()

def load_history() -> dict[str, list[float]]:
    return _load_json(HISTORY_FILE)

def load_bleu_scores() -> dict[str, float]:
    return _load_json(BLEU_SCORES_FILE)

def load_predictions(n: int = 10) -> dict[str, dict[str, str | list[str]]]:
    return _load_json(TEST_PREDICTIONS_FILE)