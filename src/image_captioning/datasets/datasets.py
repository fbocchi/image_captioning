import numpy as np
import tensorflow as tf


def create_dataset_from_tfrecords(
    tfrecord_path: str,
    image_features: dict[str, np.ndarray],
    parse_fn,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int | None = None,
):
    features = []
    inputs = []
    targets = []

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    for record in raw_dataset:
        image_id, input_caption, target_caption = parse_fn(record)
        features.append(image_features[image_id])
        inputs.append(input_caption)
        targets.append(target_caption)

    dataset = tf.data.Dataset.from_tensor_slices(
        ((features, inputs), targets)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(features),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)