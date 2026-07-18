from pathlib import Path

from image_captioning.config import (
    SPLITS_FILE,
    LEFT_TF_RECORD_FILE,
    TRAIN_TF_RECORD_FILE,
    VAL_TF_RECORD_FILE,
    TEST_TF_RECORD_FILE,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.datasets.tfrecord import (
    create_records,
    write_tfrecord,
)
from image_captioning.utils import (
    load_splits,
    load_vectorizer,
)


TF_RECORD_FILES: dict[str, Path] = {
    "train": TRAIN_TF_RECORD_FILE,
    "val": VAL_TF_RECORD_FILE,
    "test": TEST_TF_RECORD_FILE,
    "leftovers": LEFT_TF_RECORD_FILE,
}


def create_tfrecord_files(
    splits: dict[str, dict[str, list[str]]],
    vectorizer,
    output_paths: dict[str, Path],
) -> None:

    for split_name, split in splits.items():

        records = create_records(
            split=split,
            vectorizer=vectorizer,
        )

        write_tfrecord(
            records=records,
            output_path=output_paths[split_name],
            total=sum(
                len(captions)
                for captions in split.values()
            ),
            description=f"Creating {split_name}",
        )


def main() -> None:
    splits = load_splits(SPLITS_FILE)
    vectorizer = load_vectorizer(VECTORIZER_CONFIG_FILE)
    create_tfrecord_files(
        splits=splits,
        vectorizer=vectorizer,
        output_paths=TF_RECORD_FILES,
    )


if __name__ == "__main__":
    main()