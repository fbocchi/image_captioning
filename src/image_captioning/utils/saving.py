import json

from keras.callbacks import History
from keras.layers import TextVectorization

from image_captioning.config.paths import (
    TEXT_VECTORIZATION_CONFIG_FILE, HISTORY_FILE
)


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