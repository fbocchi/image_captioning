import tensorflow as tf


def create_dataset(
    tfrecord_path: str,
    image_features: dict,
    text_vec_output_sequence_length: int,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
):
    features = []
    inputs = []
    targets = []

    raw_dataset = tf.data.TFRecordDataset(
        tfrecord_path
    )

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
                            text_vec_output_sequence_length
                        ],
                        tf.int64,
                    )
                ),
                "target_caption": (
                    tf.io.FixedLenFeature(
                        [
                            text_vec_output_sequence_length
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