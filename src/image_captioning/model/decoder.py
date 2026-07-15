import tensorflow as tf

from keras import Model
from keras.layers import (
    Dropout,
    Embedding,
    LSTMCell,
)

from .deep_output import DeepOutput


@tf.keras.utils.register_keras_serializable()
class Decoder(Model):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            dropout_rate: float = 0.3,
            deep_output_dropout_rate: float = 0.2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = vocab_size
        self.m = embedding_dim
        self.n = hidden_dim

        self.dropout_rate = dropout_rate
        self.deep_output_dropout_rate = (
            deep_output_dropout_rate
        )

        self.embedding = Embedding(
            input_dim=self.K,
            output_dim=self.m,
            name="embedding",
        )

        self.dropout = Dropout(
            rate=self.dropout_rate,
            name="dropout",
        )

        self.lstm_cell = LSTMCell(
            self.n,
            name="lstm_cell",
        )

        self.deep_output = DeepOutput(
            vocab_size=self.K,
            dropout_rate=self.deep_output_dropout_rate,
            name="deep_output",
        )

    def call(
            self,
            inputs,
            training=False,
    ):
        w_t, z_t, h_t_1, c_t_1 = inputs

        e_t = self.embedding(
            w_t,
        )

        x_t = tf.concat(
            [
                e_t,
                z_t,
            ],
            axis=-1,
        )

        x_t = self.dropout(
            x_t,
            training=training,
        )

        _, [h_t, c_t] = self.lstm_cell(
            x_t,
            states=[
                h_t_1,
                c_t_1,
            ],
            training=training,
        )

        probs = self.deep_output(
            [
                h_t,
                z_t,
                e_t,
            ],
            training=training,
        )

        return probs, h_t, c_t

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
                "embedding_dim": self.m,
                "hidden_dim": self.n,
                "dropout_rate": self.dropout_rate,
                "deep_output_dropout_rate": (
                    self.deep_output_dropout_rate
                ),
            }
        )

        return config

    def compute_output_shape(
            self,
            input_shape,
    ):
        w_t_shape = input_shape[0]

        B = w_t_shape[0]

        return (
            (
                B,
                self.K,
            ),
            (
                B,
                self.n,
            ),
            (
                B,
                self.n,
            ),
        )