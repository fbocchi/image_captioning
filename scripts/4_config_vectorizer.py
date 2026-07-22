from pathlib import Path

from config import (
    END_TOKEN,
    MAX_CAPTION_LEN,
    NEW_SPLITS_FILE,
    START_TOKEN,
    VECTORIZER_CONFIG_FILE,
)
from image_captioning.utils import (
    build_vocab,
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
    captions = load_training_captions(NEW_SPLITS_FILE)

    captions = add_start_and_end_tokens(captions)

    vocabulary = build_vocab(captions)

    vectorizer = create_vectorizer(
        vocabulary=vocabulary,
        output_sequence_length=MAX_CAPTION_LEN + 2 # [START] ed [END]
    )

    print(f"Number of training captions: {len(captions)}")

    config = vectorizer.get_config()

    print(f"Vocabulary size: {config['vocabulary_size']}")
    print(f"Output sequence length: {config['output_sequence_length']}")

    save_vectorizer_config(vectorizer, to=VECTORIZER_CONFIG_FILE)


if __name__ == "__main__":
    main()
