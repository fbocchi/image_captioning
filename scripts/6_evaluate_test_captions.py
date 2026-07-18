from image_captioning.config import (
    BLEU_SCORES_FILE,
    SPLITS_FILE,
    TEST_PREDICTIONS_FILE,
)
from image_captioning.evaluation import evaluate_bleu
from image_captioning.utils import (
    load_predictions,
    load_split,
    save_bleu_scores,
)

def load_test_split() -> dict[str, list[str]]:
    return load_split(SPLITS_FILE, split_name="test")

def print_bleu_scores(scores: dict[str, float]) -> None:

    print("\n===== BLEU SCORES =====")

    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")


def main() -> None:

    predictions = load_predictions(TEST_PREDICTIONS_FILE)
    references = load_test_split()

    scores = evaluate_bleu(
        predictions=predictions,
        references=references,
    )

    print_bleu_scores(scores)

    save_bleu_scores(
        scores,
        to=BLEU_SCORES_FILE,
    )


if __name__ == "__main__":
    main()