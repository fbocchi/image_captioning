import tensorflow as tf

from keras.layers import Layer, LSTMCell


@tf.keras.utils.register_keras_serializable()
class Decoder(Layer):

    def __init__(
        self,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n = hidden_dim

        self.lstm_cell = LSTMCell(
            self.n,
            name="lstm_cell",
        )

    def call(
            self,
            #e_t: tf.Tensor,
            #z_t: tf.Tensor,
            #h_t_1: tf.Tensor,
            #c_t_1: tf.Tensor,
            inputs,
            training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:

        e_t, z_t, h_t_1, c_t_1 = inputs

        x_t = tf.concat([e_t, z_t], axis=-1)

        _, [h_t, c_t] = self.lstm_cell(
            x_t,
            states=[h_t_1, c_t_1],
            training=training,
        )

        return h_t, c_t

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.n,
            }
        )

        return config

    def compute_output_shape(
            self,
            input_shape,
    ):
        e_t_shape, _, _, _ = input_shape

        B = e_t_shape[0]

        return (
            (B, self.n),
            (B, self.n),
        )