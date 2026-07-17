from image_captioning.config import (
    END_TOKEN,
    SPLITS_FILE,
    START_TOKEN,
    VECTORIZER_CONFIG_FILE
)
from image_captioning.utils import (
    create_vectorizer,
    load_training_captions,
    save_vectorizer_config
)


def add_start_and_end_tokens(captions: list[str]) -> list[str]:
    return [
        f"{START_TOKEN} {caption} {END_TOKEN}"
        for caption in captions
    ]


def compute_max_caption_length(captions: list[str]) -> int:
    return max(
        len(caption.split())
        for caption in captions
    )


def main() -> None:
    captions = load_training_captions(SPLITS_FILE)

    captions = add_start_and_end_tokens(captions)

    vectorizer = create_vectorizer(
        captions=captions,
        output_sequence_length_fn=compute_max_caption_length
    )

    print(f"Number of training captions: {len(captions)}")

    config = vectorizer.get_config()

    print(f"Vocabulary size: {config['vocabulary_size']}")
    print(f"Output sequence length: {config['output_sequence_length']}")

    save_vectorizer_config(vectorizer, to=VECTORIZER_CONFIG_FILE)


if __name__ == "__main__":
    main()
