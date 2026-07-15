import tensorflow as tf

from keras.layers import (
    Dense,
    Dropout,
    Layer,
)


@tf.keras.utils.register_keras_serializable()
class Encoder(Layer):

    def __init__(
            self,
            output_dim: int = 256,
            dropout_rate: float = 0.2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.projection = Dense(
            units=self.output_dim,
            activation="relu",
            name="projection",
        )

        self.dropout = Dropout(
            rate=self.dropout_rate,
            name="dropout",
        )

    def call(
            self,
            feature_maps,
            training=False,
    ):
        shape = tf.shape(
            feature_maps
        )

        B = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]

        L = H * W
        D = C

        features = tf.reshape(
            feature_maps,
            shape=(
                B,
                L,
                D,
            ),
        )

        features = self.projection(
            features
        )

        features = self.dropout(
            features,
            training=training,
        )

        return features

    def compute_output_shape(
            self,
            input_shape,
    ):
        B, H, W, _ = input_shape

        L = (
            H * W
            if H is not None and W is not None
            else None
        )

        return (
            B,
            L,
            self.output_dim,
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "output_dim": self.output_dim,
                "dropout_rate": self.dropout_rate,
            }
        )

        return config