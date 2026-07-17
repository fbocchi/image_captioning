import json
from pathlib import Path

from keras.callbacks import History
from keras.layers import TextVectorization

from model import ShowAttendAndTell


def _save_json(data, to: Path) -> None:
    with to.open(mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_vectorizer_config(vectorizer: TextVectorization, to: Path) -> None:
    _save_json(vectorizer.get_config(), to=to)


def save_training_history(history: History, to: Path) -> None:
    _save_json(history.history, to=to)


def save_model(model: ShowAttendAndTell, to: Path) -> None:
    model.save(to)


def save_bleu_scores(scores: dict[str, float], to: Path) -> None:
    _save_json(scores, to=to)


def save_predictions(predictions, to: Path) -> None:
    _save_json(predictions, to=to)