import tensorflow as tf

from keras.layers import (
    Dropout,
    Layer,
)


@tf.keras.utils.register_keras_serializable()
class DeepOutput(Layer):

    def __init__(
            self,
            vocab_size: int,
            dropout_rate: float = 0.2,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = vocab_size
        self.dropout_rate = dropout_rate

        self.dropout = Dropout(
            rate=self.dropout_rate,
            name="dropout",
        )

        self.L_h = None
        self.L_z = None
        self.L_o = None
        self.b_o = None


    def build(self, input_shape):

        h_shape, z_shape, e_shape = input_shape

        m = e_shape[-1] # embedding dim
        n = h_shape[-1] # hidden dim
        D = z_shape[-1] # annotation dim

        self.L_h = self.add_weight(
            name="L_h",
            shape=(n, m),
            initializer="glorot_uniform",
            trainable=True
        )

        self.L_z = self.add_weight(
            name="L_z",
            shape=(D, m),
            initializer="glorot_uniform",
            trainable=True
        )

        self.L_o = self.add_weight(
            name="L_o",
            shape=(m, self.K),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b_o = self.add_weight(
            name="b_o",
            shape=(self.K,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)


    def call(self, inputs, training=False):

        h_t, z_t, e_t = inputs # h_t = LSTMCell([e_t; z_t], h_{t-1},...)

        # h_t L_h

        # h_t: (B,n)
        # L_h: (n,m)

        # h_t L_h: (B,m)

        h_tL_h = tf.matmul(h_t, self.L_h)

        # z_t L_z

        # z_t: (B,D)
        # L_z: (D,m)

        # z_t L_z: (B,m)

        z_tL_z = tf.matmul(
            z_t,
            self.L_z
        )

        # o_t = E(w_t) + L_h h_t + L_z z_t
        #     = e_t    + L_h h_t + L_z z_t

        # e_t: (B,m)
        # L_h h_t: (B,m)
        # L_z z_t: (B,m)

        # o_t: (B,m)

        o_t = e_t + h_tL_h + z_tL_z

        # Dropout

        o_t = self.dropout(
            o_t,
            training=training,
        )

        # L_o o_t + b_o

        # o_t: (B,m)
        # L_o: (m,K)

        # L_o o_t: (B,K)

        logits = (
            tf.matmul(
                o_t,
                self.L_o,
            )
            + self.b_o
        )

        return tf.nn.softmax(
            logits,
            axis=-1,
        )


    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
                "dropout_rate": self.dropout_rate,
            }
        )

        return config

    def compute_output_shape(
            self,
            input_shape,
    ):
        h_t_shape, _, _ = input_shape

        B = h_t_shape[0]

        return (
            B,
            self.K,
        )