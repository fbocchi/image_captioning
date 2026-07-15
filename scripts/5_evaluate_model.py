import keras


from image_captioning.config.paths import (
    BEST_MODEL_FILE,
)
from image_captioning.evaluation.evaluation import (
    evaluate_model,
)
from image_captioning.utils.loading import (
    load_features,
    load_test_split,
    load_text_vectorization,
)


MAX_CAPTION_LENGTH = 39
NUMBER_OF_EXAMPLES = 10


def main() -> None:

    model = keras.models.load_model(
        BEST_MODEL_FILE,
        compile=False,
    )

    feature_maps = load_features()

    test_split = load_test_split()

    text_vectorization = (
        load_text_vectorization()
    )

    evaluate_model(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        text_vectorization=(
            text_vectorization
        ),
        max_caption_length=(
            MAX_CAPTION_LENGTH
        ),
        number_of_examples=(
            NUMBER_OF_EXAMPLES
        ),
    )


if __name__ == "__main__":
    main()