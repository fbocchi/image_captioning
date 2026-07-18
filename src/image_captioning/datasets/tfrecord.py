from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeAlias

import tensorflow as tf
from tqdm import tqdm

from config import (
    END_TOKEN,
    START_TOKEN,
)


TFRecordExample: TypeAlias = tuple[str, list[int], list[int]]


def create_records(
    split: dict[str, list[str]],
    vectorizer: tf.keras.layers.TextVectorization,
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

        vectorized_input_captions = vectorizer(input_captions)
        vectorized_target_captions = vectorizer(target_captions)

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
            writer.write(
                serialize_example(
                    image_id=image_id,
                    input_caption=input_caption,
                    target_caption=target_caption,
                )
            )


def count_records(
    split: dict[str, list[str]],
) -> int:
    return sum(
        len(captions)
        for captions in split.values()
    )


def serialize_example(
    image_id: str,
    input_caption: list[int],
    target_caption: list[int],
) -> bytes:

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image_id": _bytes_feature(image_id),
                "input_caption": _int64_list_feature(input_caption),
                "target_caption": _int64_list_feature(target_caption),
            }
        )
    )

    return example.SerializeToString()


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