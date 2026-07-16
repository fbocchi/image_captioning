from image_captioning.utils.loading import (
    load_history, load_bleu_scores, load_predictions
)

from image_captioning.visualization import (
    plot_history,
    plot_bleu_scores,
    show_predictions,
)

from config.paths import (
    HISTORY_FILE,
    BLEU_SCORES_FILE,
    TEST_PREDICTIONS_FILE,
    HISTORY_PNG_FILE,
    BLEU_SCORES_PNG_FILE,
    TEST_PREDICTIONS_PNGS_DIR,
    IMAGES_DIR,
)


def main():

    history = load_history()

    bleu_scores = load_bleu_scores()

    predictions = load_predictions()

    plot_history(history, output_file=HISTORY_PNG_FILE)

    plot_bleu_scores(bleu_scores, output_file=BLEU_SCORES_PNG_FILE)

    show_predictions(
        predictions=predictions,
        images_dir=IMAGES_DIR,
        number_of_examples=10,
        output_dir=TEST_PREDICTIONS_PNGS_DIR
    )


if __name__ == "__main__":

    main()