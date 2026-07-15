from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeAlias

import tensorflow as tf
from tqdm import tqdm

from image_captioning.config.config import (
    END_TOKEN,
    START_TOKEN
)
from image_captioning.config.paths import (
    TEST_TF_RECORD_FILE,
    TRAIN_TF_RECORD_FILE,
    VAL_TF_RECORD_FILE,
)


# image_id, vectorized input caption, vectorized target caption
TFRecordExample: TypeAlias = tuple[
    str,
    list[int],
    list[int],
]


TF_RECORD_FILES: dict[str, Path] = {
    "train": TRAIN_TF_RECORD_FILE,
    "val": VAL_TF_RECORD_FILE,
    "test": TEST_TF_RECORD_FILE,
}


def create_tfrecord_datasets(
    splits: dict[str, dict[str, list[str]]],
    text_vectorizer: tf.keras.layers.TextVectorization,
) -> None:

    for split_name, output_path in TF_RECORD_FILES.items():

        split = splits[split_name]

        records = create_records(
            split=split,
            text_vectorizer=text_vectorizer,
        )

        write_tfrecord(
            records=records,
            output_path=output_path,
            total=count_records(split),
            description=f"Creating {split_name}",
        )

def create_records(
    split: dict[str, list[str]],
    text_vectorizer: tf.keras.layers.TextVectorization,
) -> Iterator[TFRecordExample]:

    for image_id, captions in split.items():

        input_captions = [
            f"{START_TOKEN} {caption}"
            for caption in captions
        ]

        target_captions = [
            f"{caption} {END_TOKEN}"
            for caption in captions
        ]

        vectorized_input_captions = text_vectorizer(
            tf.constant(input_captions)
        )

        vectorized_target_captions = text_vectorizer(
            tf.constant(target_captions)
        )

        for input_caption, target_caption in zip(
            vectorized_input_captions,
            vectorized_target_captions,
            strict=True,
        ):

            yield (
                image_id,
                input_caption.numpy().tolist(),
                target_caption.numpy().tolist(),
            )

def write_tfrecord(
    records: Iterable[TFRecordExample],
    output_path: Path,
    total: int,
    description: str,
) -> None:

    with tf.io.TFRecordWriter(str(output_path)) as writer:

        for (
            image_id,
            input_caption,
            target_caption,
        ) in tqdm(
            records,
            total=total,
            desc=description,
            unit="record",
        ):

            serialized_example = serialize_example(
                image_id=image_id,
                input_caption=input_caption,
                target_caption=target_caption,
            )

            writer.write(serialized_example)

def count_records(
    split: dict[str, list[str]],
) -> int:

    return sum(
        len(captions)
        for captions in split.values()
    )

def _bytes_feature(
    value: str,
) -> tf.train.Feature:

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value.encode("utf-8")]
        )
    )

def _int64_list_feature(
    values: list[int],
) -> tf.train.Feature:

    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=values
        )
    )

def serialize_example(
    image_id: str,
    input_caption: list[int],
    target_caption: list[int],
) -> bytes:

    features = {
        "image_id": _bytes_feature(
            image_id
        ),
        "input_caption": _int64_list_feature(
            input_caption
        ),
        "target_caption": _int64_list_feature(
            target_caption
        ),
    }

    example = tf.train.Example(
        features=tf.train.Features(
            feature=features
        )
    )

    return example.SerializeToString()

