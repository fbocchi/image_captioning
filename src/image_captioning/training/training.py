from tensorflow.data import Dataset
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer

from .compilation import compile_model
from .callbacks import get_callbacks

from image_captioning.config import NUM_EPOCHS, TRAINING_VERBOSE
from image_captioning.model import ShowAttendAndTell


def train_model(
        model: ShowAttendAndTell,
        train_set: Dataset,
        val_set: Dataset,
        optimizer: Optimizer,
) -> History:
    compile_model(model, optimizer)

    callbacks = get_callbacks()

    return model.fit(
        train_set,
        validation_data=val_set,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=TRAINING_VERBOSE,
        shuffle=False,
    )