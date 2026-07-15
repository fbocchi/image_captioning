from tensorflow.keras.optimizers import Adam

from image_captioning.config.config import LEARNING_RATE
from image_captioning.model import ShowAttendAndTell

from .masked_loss import masked_loss


def compile_model(
        model: ShowAttendAndTell,
) -> None:

    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss=masked_loss,
    )