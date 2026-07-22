from config import (
    BLEU_SCORES_FILE,
    BLEU_SCORES_PNG_FILE,
    HISTORY_FILE,
    HISTORY_PNG_FILE,
    IMAGES_DIR,
    TEST_PREDICTIONS_FILE,
    TEST_PREDICTIONS_PNGS_DIR,
    LR_PNG_FILE, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
)
from image_captioning.utils import (
    load_bleu_scores,
    load_history,
    load_predictions
)
from image_captioning.visualization import (
    plot_bleu_scores,
    plot_history,
    save_predictions,
)


def main():
    history = load_history(HISTORY_FILE)
    bleu_scores = load_bleu_scores(BLEU_SCORES_FILE)
    predictions = load_predictions(TEST_PREDICTIONS_FILE)

    plot_history(
        history,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        reduce_lr_on_plateau_factor=REDUCE_LR_FACTOR,
        reduce_lr_on_plateau_patience=REDUCE_LR_PATIENCE,
        loss_plot_out_file=HISTORY_PNG_FILE,
        lr_plot_out_file=LR_PNG_FILE
    )

    plot_bleu_scores(bleu_scores, output_file=BLEU_SCORES_PNG_FILE)

    save_predictions(
        predictions=predictions,
        images_dir=IMAGES_DIR,
        number_of_examples=10,
        output_dir=TEST_PREDICTIONS_PNGS_DIR
    )


if __name__ == "__main__":

    main()