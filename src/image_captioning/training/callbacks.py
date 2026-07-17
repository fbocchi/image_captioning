from keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from image_captioning.config import (
    BEST_MODEL_FILE,

    CHECKPOINT_MODE,
    CHECKPOINT_MONITOR,
    CHECKPOINT_SAVE_BEST_ONLY,
    CHECKPOINT_SAVE_FREQ,
    CHECKPOINT_SAVE_WEIGHTS_ONLY,
    CHECKPOINT_VERBOSE,

    REDUCE_LR_MONITOR,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_VERBOSE,
    REDUCE_LR_MODE,
    REDUCE_LR_MIN_DELTA,
    REDUCE_LR_MIN_LR,

    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_VERBOSE,
    EARLY_STOPPING_MODE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS
)


def get_callbacks() -> list[Callback]:
    return [
        ModelCheckpoint(
            filepath=BEST_MODEL_FILE,
            monitor=CHECKPOINT_MONITOR,
            verbose=CHECKPOINT_VERBOSE,
            save_best_only=CHECKPOINT_SAVE_BEST_ONLY,
            save_weights_only=CHECKPOINT_SAVE_WEIGHTS_ONLY,
            mode=CHECKPOINT_MODE,
            save_freq=CHECKPOINT_SAVE_FREQ
        ),
        ReduceLROnPlateau(
            monitor=REDUCE_LR_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            verbose=REDUCE_LR_VERBOSE,
            mode=REDUCE_LR_MODE,
            min_delta=REDUCE_LR_MIN_DELTA,
            min_lr=REDUCE_LR_MIN_LR
        ),
        EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            patience=EARLY_STOPPING_PATIENCE,
            verbose=EARLY_STOPPING_VERBOSE,
            mode=EARLY_STOPPING_MODE,
            restore_best_weights=EARLY_STOPPING_RESTORE_BEST_WEIGHTS
        )
    ]