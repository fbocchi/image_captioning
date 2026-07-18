import tensorflow as tf

from keras import Model
from keras.layers import Dense

from .bahdanau_attention import BahdanauAttention
from .decoder import Decoder
from .encoder import Encoder


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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = vocab_size
        self.m = decoder_embedding_dim
        self.n = decoder_hidden_dim
        self.d_att = attention_hidden_dim

        self.encoder_dim = encoder_output_dim
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.deep_output_dropout_rate = deep_output_dropout_rate

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
            deep_output_dropout_rate=self.deep_output_dropout_rate,
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

    def call(self, inputs, training=False):

        feature_maps, captions = inputs

        A = self.encoder(
            feature_maps,
            training=training,
        )

        h_t_1, c_t_1 = self.compute_initial_decoder_states(A)  # h_{-1}, c_{-1}

        outputs = []

        T = captions.shape[1]  # captions.shape = (B, T)

        for t in range(T):

            w_t = captions[:, t]

            probs, h_t, c_t, alpha_t = self.decode_step(
                A=A,
                w_t=w_t,
                h_t_1=h_t_1,
                c_t_1=c_t_1,
                training=training,
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
            [A, h_t_1],
            training=training,
        )

        probs, h_t, c_t = self.decoder(
            [w_t, z_t, h_t_1, c_t_1],
            training=training,
        )

        return probs, h_t, c_t, alpha_t

    def compute_initial_decoder_states(self, A):

        mean_A = tf.reduce_mean(
            A,
            axis=1,
        )

        return self.init_h(mean_A), self.init_c(mean_A)

    def generate_captions(
        self,
        feature_maps,
        vectorizer,
        max_caption_length: int,
    ) -> list[list[str]]:

        vocabulary = vectorizer.get_vocabulary()

        word_to_index = {
            word: index
            for index, word in enumerate(vocabulary)
        }

        start_token_id = word_to_index["[START]"]
        end_token_id = word_to_index["[END]"]

        B = feature_maps.shape[0]

        A = self.encoder(
            feature_maps,
            training=False,
        )

        h_t_1, c_t_1 = self.compute_initial_decoder_states(A)

        current_token_ids = tf.fill(
            [B],
            start_token_id,
        )

        generated_tokens = [[] for _ in range(B)]

        finished = [False] * B

        for _ in range(max_caption_length):

            probs, h_t, c_t, _ = self.decode_step(
                A=A,
                w_t=current_token_ids,
                h_t_1=h_t_1,
                c_t_1=c_t_1,
                training=False,
            )

            current_token_ids = tf.argmax(
                probs,
                axis=-1,
                output_type=tf.int32,
            )

            token_ids = current_token_ids.numpy()

            for i, token_id in enumerate(token_ids):

                if finished[i]:
                    continue

                if token_id == end_token_id:
                    finished[i] = True
                    continue

                if token_id != 0:
                    generated_tokens[i].append(
                        vocabulary[token_id]
                    )

            if all(finished):
                break

            h_t_1 = h_t
            c_t_1 = c_t

        return generated_tokens

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
                "decoder_embedding_dim": self.m,
                "decoder_hidden_dim": self.n,
                "attention_hidden_dim": self.d_att,
                "encoder_output_dim": self.encoder_dim,
                "encoder_dropout_rate": self.encoder_dropout_rate,
                "decoder_dropout_rate": self.decoder_dropout_rate,
                "deep_output_dropout_rate": self.deep_output_dropout_rate,
            }
        )

        return config

    def compute_output_shape(self, input_shape):

        _, captions_shape = input_shape

        B = captions_shape[0]
        T = captions_shape[1]

        return B, T, self.K