import tensorflow as tf

from keras import Model
from keras.layers import Dense

from .encoder import Encoder
from .decoder import Decoder
from .bahdanau_attention import BahdanauAttention


@tf.keras.utils.register_keras_serializable()
class ShowAttendAndTell(Model):

    def __init__(
            self,
            vocab_size: int,
            decoder_embedding_dim: int = 256,
            decoder_hidden_dim: int = 256,
            attention_hidden_dim: int = 256,
            encoder_output_dim: int = 256,
            encoder_dropout_rate: float = 0.2,
            decoder_dropout_rate: float = 0.3,
            deep_output_dropout_rate: float = 0.2,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.K = vocab_size
        self.m = decoder_embedding_dim
        self.n = decoder_hidden_dim
        self.d_att = attention_hidden_dim

        self.encoder_dim = encoder_output_dim
        self.encoder_dropout_rate = (
            encoder_dropout_rate
        )

        self.decoder_dropout_rate = (
            decoder_dropout_rate
        )

        self.deep_output_dropout_rate = (
            deep_output_dropout_rate
        )

        self.encoder = Encoder(
            output_dim=self.encoder_dim,
            dropout_rate=self.encoder_dropout_rate,
            name="encoder",
        )

        self.attention = BahdanauAttention(
            attention_dim=self.d_att,
            name="bahdanau_attention",
        )

        self.decoder = Decoder(
            vocab_size=self.K,
            embedding_dim=self.m,
            hidden_dim=self.n,
            dropout_rate=self.decoder_dropout_rate,
            deep_output_dropout_rate=(
                self.deep_output_dropout_rate
            ),
            name="decoder",
        )

        self.init_h = Dense(
            self.n,
            activation="tanh",
            name="init_h",
        )

        self.init_c = Dense(
            self.n,
            activation="tanh",
            name="init_c",
        )

    def call(
            self,
            inputs,
            training=False,
    ):

        feature_maps, captions = inputs

        A = self.encoder(
            feature_maps,
            training=training,
        )

        h_t_1, c_t_1 = (
            self.compute_initial_decoder_states(A)
        ) # h_{-1} e c_{-1}

        outputs = []

        T = captions.shape[1] # captions.shape = (B, T) = (B, max_caption_length)

        for t in range(T):

            w_t = captions[:, t]

            probs, h_t, c_t, alpha_t = (
                self.decode_step(
                    A=A,
                    w_t=w_t,
                    h_t_1=h_t_1,
                    c_t_1=c_t_1,
                    training=training,
                )
            )

            outputs.append(probs)

            h_t_1 = h_t
            c_t_1 = c_t

        outputs = tf.stack(
            outputs,
            axis=1,
        )

        return outputs

    def decode_step(
            self,
            A,
            w_t,
            h_t_1,
            c_t_1,
            training=False,
    ):

        z_t, alpha_t = self.attention(
            [
                A,
                h_t_1,
            ],
            training=training,
        )

        probs, h_t, c_t = self.decoder(
            [
                w_t,
                z_t,
                h_t_1,
                c_t_1,
            ],
            training=training,
        )

        return (
            probs,
            h_t,
            c_t,
            alpha_t,
        )

    def compute_initial_decoder_states(
            self,
            A,
    ):

        mean_A = tf.reduce_mean(
            A,
            axis=1,
        )

        return (
            self.init_h(mean_A),
            self.init_c(mean_A),
        )

    def generate_caption(
            self,
            feature_map,
            vectorizer,
            max_caption_length: int,
    ) -> list[str]:

        vocabulary = (
            vectorizer.get_vocabulary()
        )

        word_to_index = {
            word: index
            for index, word
            in enumerate(
                vocabulary
            )
        }

        start_token_id = (
            word_to_index[
                "[START]"
            ]
        )

        end_token_id = (
            word_to_index[
                "[END]"
            ]
        )

        feature_map = tf.convert_to_tensor(
            feature_map,
            dtype=tf.float32,
        )

        feature_map = tf.expand_dims(
            feature_map,
            axis=0,
        )

        A = self.encoder(
            feature_map,
            training=False,
        )

        h_t_1, c_t_1 = (
            self.compute_initial_decoder_states(
                A
            )
        )

        current_token_id = (
            start_token_id
        )

        generated_tokens = []

        for _ in range(
                max_caption_length
        ):

            w_t = tf.constant(
                [
                    current_token_id
                ],
                dtype=tf.int32,
            )

            probs, h_t, c_t, _ = (
                self.decode_step(
                    A=A,
                    w_t=w_t,
                    h_t_1=h_t_1,
                    c_t_1=c_t_1,
                    training=False,
                )
            )

            current_token_id = int(
                tf.argmax(
                    probs[0],
                    axis=-1,
                ).numpy()
            )

            if (
                    current_token_id
                    == end_token_id
            ):
                break

            if current_token_id != 0:
                generated_tokens.append(
                    vocabulary[
                        current_token_id
                    ]
                )

            h_t_1 = h_t
            c_t_1 = c_t

        return generated_tokens

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
                "embedding_output_dim": self.m,
                "decoder_hidden_dim": self.n,
                "attention_hidden_dim": self.d_att,
                "encoder_output_dim": self.encoder_dim,
                "encoder_dropout_rate": (
                    self.encoder_dropout_rate
                ),
                "decoder_dropout_rate": (
                    self.decoder_dropout_rate
                ),
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
        feature_maps_shape, captions_shape = (
            input_shape
        )

        B = captions_shape[0]
        T = captions_shape[1]

        return (
            B,
            T,
            self.K,
        )