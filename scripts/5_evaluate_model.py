import keras


from image_captioning.config import (
    BEST_MODEL_FILE,
    BLEU_SCORES_FILE,
    FEATURES_FILE,
    SPLITS_FILE,
    TEST_PREDICTIONS_FILE,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.evaluation import evaluate_model
from image_captioning.utils import (
    load_features,
    load_test_split,
    load_vectorizer,
    save_bleu_scores,
    save_predictions
)


MAX_CAPTION_LENGTH = 39
NUMBER_OF_EXAMPLES = 10


def print_bleu_scores(scores: dict[str, float]) -> None:
    print("\n===== BLEU SCORES =====")
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")


def print_generated_captions(
    captions: dict[str, dict[str, str | list[str]]],
    number_of_examples: int,
) -> None:
    print("\n===== GENERATED CAPTIONS =====")
    for image_id, data in list(captions.items())[:number_of_examples]:
        print(f"\nImage: {image_id}")
        print("Generated:", data["generated"])
        print("References:")
        for reference in data["reference"]:
            print(f"- {reference}")


def main() -> None:
    model = keras.models.load_model(BEST_MODEL_FILE, compile=False)

    feature_maps = load_features(FEATURES_FILE)
    test_split = load_test_split(SPLITS_FILE)
    text_vectorization = load_vectorizer(VECTORIZER_CONFIG_FILE)

    bleu_scores, captions = evaluate_model(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        text_vectorization=text_vectorization,
        max_caption_length=MAX_CAPTION_LENGTH,
    )

    print_bleu_scores(bleu_scores)
    print_generated_captions(captions, number_of_examples=NUMBER_OF_EXAMPLES)

    save_bleu_scores(bleu_scores, to=BLEU_SCORES_FILE)
    save_predictions(captions, to=TEST_PREDICTIONS_FILE)


if __name__ == "__main__":
    main()