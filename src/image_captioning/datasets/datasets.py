from collections.abc import Iterator

import numpy as np
import tensorflow as tf


def tfrecord_generator(
    tfrecord_path: str,
    image_features: dict[str, np.ndarray],
    text_vec_output_sequence_length: int,
) -> Iterator[
    tuple[
        tuple[
            np.ndarray,
            np.ndarray,
        ],
        np.ndarray,
    ]
]:
    feature_description = {
        "image_id": tf.io.FixedLenFeature(
            shape=[],
            dtype=tf.string,
        ),
        "input_caption": tf.io.FixedLenFeature(
            shape=(
                text_vec_output_sequence_length,
            ),
            dtype=tf.int64,
        ),
        "target_caption": tf.io.FixedLenFeature(
            shape=(
                text_vec_output_sequence_length,
            ),
            dtype=tf.int64,
        ),
    }

    raw_dataset = tf.data.TFRecordDataset(
        tfrecord_path
    )

    for serialized_example in raw_dataset:
        example = tf.io.parse_single_example(
            serialized_example,
            feature_description,
        )

        image_id = (
            example["image_id"]
            .numpy()
            .decode("utf-8")
        )

        image_feature = (
            image_features[image_id]
            .astype(
                np.float32,
                copy=False,
            )
        )

        input_caption = (
            example["input_caption"]
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        )

        target_caption = (
            example["target_caption"]
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        )

        yield (
            (
                image_feature,
                input_caption,
            ),
            target_caption,
        )


def create_dataset(
    tfrecord_path: str,
    image_features: dict[str, np.ndarray],
    text_vec_output_sequence_length: int,
    batch_size: int = 64,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int | None = None,
) -> tf.data.Dataset:

    image_feature_shape = next(
        iter(image_features.values())
    ).shape

    dataset = tf.data.Dataset.from_generator(
        tfrecord_generator,
        args=(
            tfrecord_path,
            image_features,
            text_vec_output_sequence_length,
        ),
        output_signature=(
            (
                tf.TensorSpec(
                    shape=image_feature_shape,
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(
                        text_vec_output_sequence_length,
                    ),
                    dtype=tf.int32,
                ),
            ),
            tf.TensorSpec(
                shape=(
                    text_vec_output_sequence_length,
                ),
                dtype=tf.int32,
            ),
        ),
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
