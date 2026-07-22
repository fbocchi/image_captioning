from collections import Counter
from typing import Callable

from keras.layers import TextVectorization


def build_vocab(captions: list[str], min_freq: int = 3) -> list[str]:

    freq_counter = Counter()

    for caption in captions:
        freq_counter.update(caption.split())

    vocab = [
        word
        for word, frequency
        in freq_counter.most_common()
        if frequency >= min_freq
    ]

    return vocab


def compute_max_caption_length(captions: list[str]) -> int:
    return max(
        len(caption.split())
        for caption in captions
    )


def compute_vectorizer_output_sequence_length(
        captions: list[str],
        output_sequence_length_fn: Callable[[list[str]], int]
) -> int:
    return output_sequence_length_fn(captions)


def create_vectorizer(
        vocabulary: list[str],
        output_sequence_length: int
) -> TextVectorization:

    return TextVectorization(
        max_tokens=None,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=output_sequence_length,
        pad_to_max_tokens=False,
        vocabulary=vocabulary,
    )
