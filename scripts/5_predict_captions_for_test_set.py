import numpy as np
import keras

from image_captioning.config import (
    BEST_MODEL_FILE,
    FEATURES_FILE,
    SPLITS_FILE,
    TEST_PREDICTIONS_FILE,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.prediction import generate_captions
from image_captioning.utils import (
    load_features,
    load_test_split,
    load_vectorizer,
    save_predictions,
)


MAX_CAPTION_LENGTH = 39


def predict_captions_for_test_set(
    model,
    test_split: dict[str, list[str]],
    feature_maps: dict[str, np.ndarray],
    vectorizer: keras.layers.TextVectorization,
    max_caption_length: int,
) -> dict[str, str]:

    image_ids = []
    feature_batch = []

    for image_id in test_split:
        image_ids.append(image_id)
        feature_batch.append(feature_maps[image_id])

    feature_batch = np.stack(feature_batch)

    generated_captions = generate_captions(
        model=model,
        image_feature_maps=feature_batch,
        vectorizer=vectorizer,
        max_caption_length=max_caption_length,
    )

    return {
        image_id: " ".join(caption)
        for image_id, caption in zip(image_ids, generated_captions)
    }


def main() -> None:

    model = keras.models.load_model(
        BEST_MODEL_FILE,
        compile=False,
    )

    feature_maps = load_features(FEATURES_FILE)
    test_split = load_test_split(SPLITS_FILE)
    vectorizer = load_vectorizer(VECTORIZER_CONFIG_FILE)

    predictions = predict_captions_for_test_set(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        vectorizer=vectorizer,
        max_caption_length=MAX_CAPTION_LENGTH
    )

    save_predictions(
        predictions,
        to=TEST_PREDICTIONS_FILE,
    )


if __name__ == "__main__":
    main()