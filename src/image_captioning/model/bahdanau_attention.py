import tensorflow as tf

from keras.layers import Layer


@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(Layer):

    def __init__(self, attention_dim, **kwargs):
        super().__init__(**kwargs)

        self.d_att = attention_dim

        self.W_a = None
        self.W_h = None
        self.b = None
        self.v = None

    def build(self, input_shape):

        A_shape, h_t_1_shape = input_shape

        D = A_shape[-1] # annotation dim
        n = h_t_1_shape[-1] # hidden dimension

        self.W_a = self.add_weight(
            name="W_a",
            shape=(D, self.d_att),
            initializer="glorot_uniform",
            trainable=True
        )

        self.W_h = self.add_weight(
            name="W_h",
            shape=(n, self.d_att),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="b",
            shape=(self.d_att,),
            initializer="zeros",
            trainable=True
        )

        self.v = self.add_weight(
            name="v",
            shape=(self.d_att, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(
            self,
            inputs,
            training=False,
    ):

        A, h_t_1 = inputs

        # U_a: A W_a

        # A: (B,L,D)
        # W_a: (D,d_att)

        # U_a: (B,L,d_att)

        U_a = tf.matmul(A, self.W_a)

        # u_t = h_t_1 W_h

        # h_t_1: (B,n)
        # W_h: (n,d_att)

        # u_h: (B,d_att)

        u_h = tf.matmul(h_t_1, self.W_h)

        # u_h: (B,d_att) -> (B,1,d_att)

        U_h = tf.expand_dims(u_h, axis=1)

        # U_t = tanh(U_a + U_h + b)

        # U_a: (B,L,d_att)
        # U_h: (B,1,d_att)
        # b: (d_att,)

        # U_t: (B,L,d_att)

        U_t = tf.tanh(U_a + U_h + self.b)

        # E_t = U_t v

        # U_t: (B,L,d_att)
        # v: (d_att,1)

        # E_t: (B,L,1)

        E_t = tf.matmul(U_t, self.v)

        # E_t: (B,L,1) -> (B,L)

        E_t = tf.squeeze(E_t, axis=-1)

        # alpha_t = softmax(E_t)

        # E_t: (B,L)

        # alpha_t: (B,L)

        alpha_t = tf.nn.softmax(E_t, axis=1)

        # alpha_t: (B, L) -> (B, 1, L)

        alpha_t_exp = tf.expand_dims(alpha_t, axis=1)

        # z_t = alpha_t_exp A

        # alpha_t_exp: (B,1,L)
        # A: (B,L,D)

        # z_t: (B,1,D)

        z_t = tf.matmul(alpha_t_exp, A)

        # z_t: (B,1,D) -> (B,D)

        z_t = tf.squeeze(z_t, axis=1)

        return z_t, alpha_t

    def get_config(self):
        config = super().get_config()
        config.update({ "attention_dim": self.d_att })
        return config

    def compute_output_shape(
            self,
            input_shape,
    ):
        A_shape, _ = input_shape

        B = A_shape[0]
        L = A_shape[1]
        D = A_shape[2]

        return (
            (
                B,
                D,
            ),
            (
                B,
                L,
            ),
        )