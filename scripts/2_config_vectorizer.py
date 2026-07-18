from pathlib import Path

from image_captioning.config import (
    END_TOKEN,
    SPLITS_FILE,
    START_TOKEN,
    VECTORIZER_CONFIG_FILE
)
from image_captioning.utils import (
    build_vocab,
    compute_max_caption_length,
    compute_vectorizer_output_sequence_length,
    create_vectorizer,
    load_split_captions,
    save_vectorizer_config
)


def load_training_captions(path: Path) -> list[str]:
    return load_split_captions(path, "train")


def add_start_and_end_tokens(captions: list[str]) -> list[str]:
    return [
        f"{START_TOKEN} {caption} {END_TOKEN}"
        for caption in captions
    ]


def main() -> None:
    captions = load_training_captions(SPLITS_FILE)

    captions = add_start_and_end_tokens(captions)

    vocabulary = build_vocab(captions)

    length = compute_vectorizer_output_sequence_length(
        captions,
        compute_max_caption_length
    )

    vectorizer = create_vectorizer(
        vocabulary=vocabulary,
        output_sequence_length=length
    )

    print(f"Number of training captions: {len(captions)}")

    config = vectorizer.get_config()

    print(f"Vocabulary size: {config['vocabulary_size']}")
    print(f"Output sequence length: {config['output_sequence_length']}")

    save_vectorizer_config(vectorizer, to=VECTORIZER_CONFIG_FILE)


if __name__ == "__main__":
    main()
