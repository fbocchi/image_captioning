from pathlib import Path

from image_captioning.config import (
    SPLITS_FILE,
    VECTORIZER_CONFIG_FILE,
    TRAIN_TF_RECORD_FILE,
    TEST_TF_RECORD_FILE,
    VAL_TF_RECORD_FILE
)
from image_captioning.datasets.tfrecord import create_tfrecord_datasets
from image_captioning.utils import load_splits, load_vectorizer


TF_RECORD_FILES: dict[str, Path] = {
    "train": TRAIN_TF_RECORD_FILE,
    "val": VAL_TF_RECORD_FILE,
    "test": TEST_TF_RECORD_FILE,
}


def main():
    splits = load_splits(SPLITS_FILE)
    vectorizer = load_vectorizer(VECTORIZER_CONFIG_FILE)
    create_tfrecord_datasets(
        splits=splits,
        vectorizer=vectorizer,
        output_paths=TF_RECORD_FILES
    )


if __name__ == "__main__":
    main()