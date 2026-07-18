from tensorflow.keras.optimizers import Optimizer

from .masked_loss import masked_loss

from image_captioning.model import ShowAttendAndTell


def compile_model(
        model: ShowAttendAndTell,
        optimizer: Optimizer
) -> None:

    model.compile(optimizer=optimizer, loss=masked_loss)