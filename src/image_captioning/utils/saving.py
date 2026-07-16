import json

from keras.callbacks import History
from keras.layers import TextVectorization

from image_captioning.config.paths import (
    TEXT_VECTORIZATION_CONFIG_FILE, HISTORY_FILE, FINAL_MODEL_FILE,
    BLEU_SCORES_FILE, TEST_PREDICTIONS_FILE
)
from model import ShowAttendAndTell


def save_text_vec_config(
    text_vec: TextVectorization,
) -> None:

    with TEXT_VECTORIZATION_CONFIG_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            text_vec.get_config(),
            file,
            indent=4,
        )

def save_training_history(
    history: History,
) -> None:

    with HISTORY_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history.history,
            file,
            indent=4,
        )

def save_model(model: ShowAttendAndTell):
    model.save(FINAL_MODEL_FILE)

def save_bleu_scores(scores):

    with BLEU_SCORES_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scores,
            file,
            indent=4,
        )

def save_predictions(predictions):

    with TEST_PREDICTIONS_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            predictions,
            file,
            indent=4,
        )