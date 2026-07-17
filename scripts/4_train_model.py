import tensorflow as tf

from config import VECTORIZER_CONFIG_FILE
from image_captioning.config import (
    RANDOM_SEED, BATCH_SIZE,
    ENCODER_OUTPUT_DIM, ENCODER_DROPOUT,
    DECODER_EMBEDDING_DIM, DECODER_HIDDEN_DIM, DECODER_DROPOUT,
    DEEP_OUTPUT_DROPOUT,
    ATTENTION_HIDDEN_DIM,

    VECTORIZER_CONFIG_FILE, FEATURES_FILE,
    HISTORY_FILE, FINAL_MODEL_FILE
)
from image_captioning.config.paths import TRAIN_TF_RECORD_FILE, VAL_TF_RECORD_FILE
from image_captioning.datasets.datasets import create_dataset
from image_captioning.model import ShowAttendAndTell
from image_captioning.training import train_model
from image_captioning.utils.loading import load_vectorizer_config, load_features
from image_captioning.utils.saving import save_training_history, save_model


def load_vec_vocab_size() -> int:
    return load_vectorizer_config(VECTORIZER_CONFIG_FILE)["vocabulary_size"]

def load_vec_output_sequence_length() -> int:
    return load_vectorizer_config(VECTORIZER_CONFIG_FILE)["output_sequence_length"]

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

    train_set = create_dataset(
        tfrecord_path=TRAIN_TF_RECORD_FILE,
        image_features=features,
        text_vec_output_sequence_length=output_sequence_length,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_set = create_dataset(
        tfrecord_path=VAL_TF_RECORD_FILE,
        image_features=features,
        text_vec_output_sequence_length=output_sequence_length,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    history = train_model(
        model,
        train_set,
        val_set=val_set,
    )

    save_training_history(history, to=HISTORY_FILE)

    save_model(model, to=FINAL_MODEL_FILE)


if __name__ == "__main__":
    main()