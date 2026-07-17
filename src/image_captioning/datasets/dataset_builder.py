import numpy as np
from pathlib import Path

import tensorflow as tf


class DatasetBuilder:

    def __init__(self, max_caption_length: int) -> None:
        self.max_caption_length = max_caption_length

    def from_tfrecord(
            self,
            path: Path,
            image_features: dict[str, np.ndarray],
            batch_size: int = 32,
            shuffle: bool = True,
            seed: int | None = None,
    ) -> tf.data.Dataset:

        features = []
        inputs = []
        targets = []

        raw_dataset = tf.data.TFRecordDataset(path)

        for record in raw_dataset:
            example = tf.io.parse_single_example(
                record,
                {
                    "image_id": (
                        tf.io.FixedLenFeature(
                            [],
                            tf.string,
                        )
                    ),
                    "input_caption": (
                        tf.io.FixedLenFeature(
                            [
                                self.max_caption_length
                            ],
                            tf.int64,
                        )
                    ),
                    "target_caption": (
                        tf.io.FixedLenFeature(
                            [
                                self.max_caption_length
                            ],
                            tf.int64,
                        )
                    ),
                },
            )

            image_id = (
                example[
                    "image_id"
                ]
                .numpy()
                .decode()
            )

            features.append(
                image_features[
                    image_id
                ]
            )

            inputs.append(
                example[
                    "input_caption"
                ].numpy()
            )

            targets.append(
                example[
                    "target_caption"
                ].numpy()
            )

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    (
                        features,
                        inputs,
                    ),
                    targets,
                )
            )
        )

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(
                    features
                ),
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