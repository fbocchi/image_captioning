import tensorflow as tf

from keras.layers import Layer


@tf.keras.utils.register_keras_serializable()
class DeepOutput(Layer):

    def __init__(
        self,
        vocab_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = vocab_size

        self.L_h = None
        self.L_z = None
        self.L_o = None
        self.b_o = None

    def build(self, input_shape) -> None:

        h_shape, z_shape, e_shape = input_shape

        m = e_shape[-1]  # embedding dim
        n = h_shape[-1]  # hidden dim
        D = z_shape[-1]  # annotation dim

        self.L_h = self.add_weight(
            name="L_h",
            shape=(n, m),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.L_z = self.add_weight(
            name="L_z",
            shape=(D, m),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.L_o = self.add_weight(
            name="L_o",
            shape=(m, self.K),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b_o = self.add_weight(
            name="b_o",
            shape=(self.K,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(
            self,
            #h_t: tf.Tensor,
            #z_t: tf.Tensor,
            #e_t: tf.Tensor,
            inputs
    ) -> tf.Tensor:

        h_t, z_t, e_t = inputs

        h_tL_h = tf.matmul(h_t, self.L_h)

        z_tL_z = tf.matmul(z_t, self.L_z)

        o_t = e_t + h_tL_h + z_tL_z

        logits = tf.matmul(o_t, self.L_o) + self.b_o

        return tf.nn.softmax(logits, axis=-1)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
            }
        )

        return config

    def compute_output_shape(
        self,
        input_shape,
    ):
        h_t_shape, _, _ = input_shape
        B = h_t_shape[0]
        return B, self.K