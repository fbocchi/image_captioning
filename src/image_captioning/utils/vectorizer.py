from collections import Counter
from collections.abc import Callable

from keras.layers import TextVectorization

from image_captioning.config import MIN_FREQ


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


def create_vectorizer(
        captions: list[str],
        output_sequence_length_fn: Callable[[list[str]], int]
) -> TextVectorization:
    vocab = build_vocab(captions, min_freq=MIN_FREQ)

    output_sequence_length = output_sequence_length_fn(captions)

    return TextVectorization(
        max_tokens=None,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=output_sequence_length,
        pad_to_max_tokens=False,
        vocabulary=vocab,
    )
