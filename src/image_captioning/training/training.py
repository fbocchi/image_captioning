from tensorflow.data import Dataset
from tensorflow.keras.callbacks import History

from .compilation import compile_model
from .callbacks import get_callbacks

from image_captioning.config.config import NUM_EPOCHS, TRAINING_VERBOSE
from image_captioning.model import ShowAttendAndTell


def train_model(
    model: ShowAttendAndTell,
    train_set: Dataset,
    val_set: Dataset,
    train_steps: int,
    val_steps: int,
) -> History:

    compile_model(model)

    callbacks = get_callbacks()


    return model.fit(
        train_set,
        steps_per_epoch=train_steps,
        validation_data=val_set,
        validation_steps=val_steps,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=TRAINING_VERBOSE,
        shuffle=False,
    )