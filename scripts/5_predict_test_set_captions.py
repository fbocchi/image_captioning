import numpy as np
import keras

from config import (
    BEST_MODEL_FILE,
    FEATURES_FILE,
    SPLITS_FILE,
    TEST_PREDICTIONS_FILE,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.prediction import generate_captions
from image_captioning.utils import (
    load_split_features,
    load_vectorizer,
    save_predictions,
)


MAX_CAPTION_LENGTH = 39


def load_test_image_features() -> dict[str, np.ndarray]:
    return load_split_features(
        features_path=FEATURES_FILE,
        split_path=SPLITS_FILE,
        split_name="test",
    )


def predict_captions(
    model,
    feature_maps: dict[str, np.ndarray],
    vectorizer: keras.layers.TextVectorization,
    max_caption_length: int,
) -> dict[str, str]:

    image_ids = list(feature_maps.keys())
    feature_batch = np.stack([
        feature_maps[image_id]
        for image_id in image_ids
    ])

    generated_captions = generate_captions(
        model=model,
        image_feature_maps=feature_batch,
        vectorizer=vectorizer,
        max_caption_length=max_caption_length,
    )

    return {
        image_id: " ".join(caption)
        for image_id, caption in zip(
            image_ids,
            generated_captions,
            strict=True,
        )
    }


def main() -> None:

    model = keras.models.load_model(
        BEST_MODEL_FILE,
        compile=False,
    )

    feature_maps = load_test_image_features()
    vectorizer = load_vectorizer(VECTORIZER_CONFIG_FILE)

    predictions = predict_captions(
        model=model,
        feature_maps=feature_maps,
        vectorizer=vectorizer,
        max_caption_length=MAX_CAPTION_LENGTH,
    )

    save_predictions(
        predictions,
        to=TEST_PREDICTIONS_FILE,
    )


if __name__ == "__main__":
    main()