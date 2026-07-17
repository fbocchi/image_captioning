import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from image_captioning.config import (
    ATTENTION_HIDDEN_DIM,
    BATCH_SIZE,
    DECODER_DROPOUT,
    DECODER_EMBEDDING_DIM,
    DECODER_HIDDEN_DIM,
    DEEP_OUTPUT_DROPOUT,
    ENCODER_OUTPUT_DIM,
    ENCODER_DROPOUT,
    FEATURES_FILE,
    FINAL_MODEL_FILE,
    HISTORY_FILE,
    LEARNING_RATE,
    RANDOM_SEED,
    TRAIN_TF_RECORD_FILE,
    VAL_TF_RECORD_FILE,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.datasets import create_dataset_from_tfrecords
from image_captioning.model import ShowAttendAndTell
from image_captioning.training import train_model
from image_captioning.utils import (
    load_features,
    load_vectorizer_config,
    save_model,
    save_training_history
)


def load_vec_vocab_size() -> int:
    return load_vectorizer_config(
        VECTORIZER_CONFIG_FILE
    )["vocabulary_size"]


def load_vec_output_sequence_length() -> int:
    return load_vectorizer_config(
        VECTORIZER_CONFIG_FILE
    )["output_sequence_length"]


def parse_caption_record(record: tf.Tensor, captions_length: int):
    example = tf.io.parse_single_example(
        record,
        {
            "image_id": tf.io.FixedLenFeature([], tf.string),
            "input_caption": tf.io.FixedLenFeature([captions_length], tf.int64),
            "target_caption": tf.io.FixedLenFeature([captions_length], tf.int64),
        },
    )

    return (
        example["image_id"].numpy().decode(),
        example["input_caption"].numpy(),
        example["target_caption"].numpy(),
    )


def create_train_and_val_sets(
        image_features: dict[str, np.ndarray],
        captions_length: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_set = create_dataset_from_tfrecords(
        TRAIN_TF_RECORD_FILE,
        image_features=image_features,
        parse_fn=lambda record: parse_caption_record(
            record,
            captions_length,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_set = create_dataset_from_tfrecords(
        VAL_TF_RECORD_FILE,
        image_features=image_features,
        parse_fn=lambda record: parse_caption_record(
            record,
            captions_length,
        ),
        batch_size=BATCH_SIZE,
    )

    return train_set, val_set


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    vocab_size = load_vec_vocab_size()
    output_sequence_length = load_vec_output_sequence_length()

    features = load_features(FEATURES_FILE)

    model = ShowAttendAndTell(
        vocab_size=vocab_size,

        encoder_output_dim=ENCODER_OUTPUT_DIM,
        encoder_dropout_rate=ENCODER_DROPOUT,

        decoder_embedding_dim=DECODER_EMBEDDING_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        decoder_dropout_rate=DECODER_DROPOUT,

        deep_output_dropout_rate=DEEP_OUTPUT_DROPOUT,

        attention_hidden_dim=ATTENTION_HIDDEN_DIM,
    )

    train_set, val_set = create_train_and_val_sets(
        features,
        captions_length=output_sequence_length
    )

    history = train_model(
        model,
        train_set,
        val_set=val_set,
        optimizer=Adam(LEARNING_RATE)
    )

    save_training_history(history, to=HISTORY_FILE)

    save_model(model, to=FINAL_MODEL_FILE)


if __name__ == "__main__":
    main()