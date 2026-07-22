import numpy as np
import tensorflow as tf

from keras import Model
from keras.layers import Dense, Dropout, Embedding, TextVectorization

from .bahdanau_attention import BahdanauAttention
from .decoder import Decoder
from .deep_output import DeepOutput
from .encoder import Encoder


@tf.keras.utils.register_keras_serializable()
class ShowAttendAndTell(Model):

    def __init__(
            self,
            vocab_size: int,
            attention_hidden_dim: int = 256,
            embedding_dim: int = 256,
            embedding_output_dropout_rate: float = 0.2,
            decoder_hidden_dim: int = 256,
            decoder_output_dropout_rate: float = 0.3,
            encoder_output_dim: int = 256,
            attention_regularization_weight: float = 0.001,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.K = vocab_size
        self.m = embedding_dim
        self.n = decoder_hidden_dim
        self.d_att = attention_hidden_dim

        self.encoder_output_dim = encoder_output_dim

        self.embedding_output_dropout_rate = embedding_output_dropout_rate
        self.decoder_output_dropout_rate = decoder_output_dropout_rate

        self.attention_regularization_weight = attention_regularization_weight

        self.encoder = Encoder(
            output_dim=self.encoder_output_dim,
            name="encoder",
        )

        self.attention = BahdanauAttention(
            attention_dim=self.d_att,
            name="bahdanau_attention",
        )

        self.embedding = Embedding(
            input_dim=self.K,
            output_dim=self.m,
            name="embedding",
        )

        self.embedding_output_dropout = Dropout(
            rate=self.embedding_output_dropout_rate,
            name="embedding_output_dropout",
        )

        self.decoder = Decoder(
            hidden_dim=self.n,
            name="decoder",
        )

        self.decoder_output_dropout = Dropout(
            rate=self.decoder_output_dropout_rate,
            name="decoder_output_dropout",
        )

        self.deep_output = DeepOutput(
            vocab_size=self.K,
            name="deep_output",
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
            training: bool = False,
    ) -> tf.Tensor:

        feature_maps, captions = inputs

        A = self.encoder(feature_maps)

        h_t_1, c_t_1 = self.compute_initial_decoder_states(A)

        outputs = []
        attention_weights = []

        T = captions.shape[1]

        for t in range(T):

            w_t = captions[:, t]

            e_t = self.embedding(w_t)

            e_t = self.embedding_output_dropout(
                e_t,
                training=training,
            )

            z_t, alpha_t, h_t, c_t = self.decode_step(
                A=A,
                e_t=e_t,
                h_t_1=h_t_1,
                c_t_1=c_t_1,
                training=training,
            )

            h_t_drop = self.decoder_output_dropout(
                h_t,
                training=training,
            )

            probs = self.deep_output(
                (h_t_drop,
                z_t,
                e_t),
            )

            outputs.append(probs)
            attention_weights.append(alpha_t)

            h_t_1 = h_t
            c_t_1 = c_t

        outputs = tf.stack(outputs, axis=1)
        attention_weights = tf.stack(attention_weights, axis=1)

        if training:
            caption_mask = tf.cast(
                captions != 0,
                attention_weights.dtype,
            )

            caption_mask = caption_mask[..., None]

            masked_attention_weights = (
                    attention_weights * caption_mask
            )

            attention_loss = self.compute_attention_regularization_loss(
                masked_attention_weights,
            )

            self.add_loss(
                tf.cast(
                    self.attention_regularization_weight,
                    attention_loss.dtype,
                ) * attention_loss
            )

        return outputs

    def compute_initial_decoder_states(
            self,
            A: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:

        mean_A = tf.reduce_mean(
            A,
            axis=1,
        )

        return (
            self.init_h(mean_A),
            self.init_c(mean_A),
        )

    def decode_step(
            self,
            A: tf.Tensor,
            e_t: tf.Tensor,
            h_t_1: tf.Tensor,
            c_t_1: tf.Tensor,
            training: bool = False,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        z_t, alpha_t = self.attention((A, h_t_1))

        h_t, c_t = self.decoder(
            (e_t,
            z_t,
            h_t_1,
            c_t_1),
            training=training,
        )

        return z_t, alpha_t, h_t, c_t

    @staticmethod
    def compute_attention_regularization_loss(
            attention_weights: tf.Tensor,
    ) -> tf.Tensor:

        attention_per_region = tf.reduce_sum(
            attention_weights,
            axis=1,
        )

        penalty = tf.square(
            1.0 - attention_per_region
        )

        return tf.reduce_mean(
            tf.reduce_sum(
                penalty,
                axis=1,
            )
        )

    def generate_captions(
            self,
            feature_maps: np.ndarray,
            vectorizer: TextVectorization,
            max_caption_length: int,
    ) -> list[list[str]]:
        vocabulary = vectorizer.get_vocabulary()

        word_to_index = {
            word: index
            for index, word in enumerate(vocabulary)
        }

        generated_token_ids = self.generate_caption_ids(
            feature_maps=feature_maps,
            start_token_id=word_to_index["[START]"],
            end_token_id=word_to_index["[END]"],
            max_caption_length=max_caption_length,
        )

        return [
            [
                vocabulary[token_id]
                for token_id in caption
            ]
            for caption in generated_token_ids
        ]

    def generate_caption_ids(
            self,
            feature_maps: np.ndarray,
            start_token_id: int,
            end_token_id: int,
            max_caption_length: int,
    ) -> list[list[int]]:

        B = feature_maps.shape[0]

        A = self.encoder(feature_maps)

        h_t_1, c_t_1 = self.compute_initial_decoder_states(A)

        current_token_ids = tf.fill(
            [B],
            start_token_id,
        )

        generated_token_ids = [[] for _ in range(B)]

        finished = [False] * B

        for _ in range(max_caption_length):

            e_t = self.embedding(current_token_ids)

            z_t, _, h_t, c_t = self.decode_step(
                A=A,
                e_t=e_t,
                h_t_1=h_t_1,
                c_t_1=c_t_1,
                training=False,
            )

            probs = self.deep_output(
                (h_t,
                z_t,
                e_t),
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
                    generated_token_ids[i].append(token_id)

            if all(finished):
                break

            h_t_1 = h_t
            c_t_1 = c_t

        return generated_token_ids

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "vocab_size": self.K,
                "attention_hidden_dim": self.d_att,
                "embedding_dim": self.m,
                "decoder_hidden_dim": self.n,
                "encoder_output_dim": self.encoder_output_dim,
                "embedding_output_dropout_rate": self.embedding_output_dropout_rate,
                "decoder_output_dropout_rate": self.decoder_output_dropout_rate,
                "attention_regularization_weight": self.attention_regularization_weight,
            }
        )

        return config

    def compute_output_shape(self, input_shape):

        _, captions_shape = input_shape

        B = captions_shape[0]
        T = captions_shape[1]

        return B, T, self.K