from image_captioning.config import (
    BLEU_SCORES_FILE,
    SPLITS_FILE,
    TEST_PREDICTIONS_FILE,
)
from image_captioning.evaluation import evaluate_test_captions
from image_captioning.utils import (
    load_predictions,
    load_test_split,
    save_bleu_scores,
)


def print_bleu_scores(scores: dict[str, float]) -> None:

    print("\n===== BLEU SCORES =====")

    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")


def main() -> None:

    predictions = load_predictions(TEST_PREDICTIONS_FILE)
    test_split = load_test_split(SPLITS_FILE)

    scores = evaluate_test_captions(
        predictions=predictions,
        test_split=test_split,
    )

    print_bleu_scores(scores)

    save_bleu_scores(
        scores,
        to=BLEU_SCORES_FILE,
    )


if __name__ == "__main__":
    main()