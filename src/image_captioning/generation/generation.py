import numpy as np
from keras.layers import TextVectorization
import tensorflow as tf

from model import ShowAttendAndTell


def generate_caption(
        model: ShowAttendAndTell,
        image_feature_map: np.ndarray,
        vectorizer: TextVectorization,
        max_caption_length: int,
) -> list[str]:

    if image_feature_map.ndim != 3:
        raise ValueError(
            "image_feature_map must have shape (14, 14, 512). "
            f"Received shape {image_feature_map.shape}."
        )

    image_feature_map = tf.expand_dims(image_feature_map, axis=0)

    return model.generate_captions(
        image_feature_map,
        vectorizer,
        max_caption_length
    )[0]


def generate_captions(
        model: ShowAttendAndTell,
        image_feature_maps: np.ndarray,
        vectorizer: TextVectorization,
        max_caption_length: int,
) -> list[list[str]]:

    if image_feature_maps.ndim != 4:
        raise ValueError(
            "image_feature_maps must have shape (B, 14, 14, 512). "
            f"Received shape {image_feature_maps.shape}."
        )

    return model.generate_captions(
        image_feature_maps,
        vectorizer,
        max_caption_length
    )