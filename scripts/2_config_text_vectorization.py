from image_captioning.config.config import END_TOKEN, START_TOKEN
from image_captioning.utils.loading import load_training_captions
from image_captioning.utils.text_vectorization import create_vectorizer
from image_captioning.utils.saving import save_text_vec_config


def add_start_and_end_tokens(
    captions: list[str],
) -> list[str]:
    return [
        f"{START_TOKEN} {caption} {END_TOKEN}"
        for caption in captions
    ]


def compute_max_caption_length(
    captions: list[str],
) -> int:
    return max(
        len(caption.split())
        for caption in captions
    )


def main() -> None:
    training_captions = load_training_captions()

    training_captions = add_start_and_end_tokens(
        training_captions
    )

    text_vectorizer = create_vectorizer(
        captions=training_captions,
        output_sequence_length_fn=(
            compute_max_caption_length
        ),
    )

    print(
        f"Number of training captions: "
        f"{len(training_captions)}"
    )

    print(
        f"Vocabulary size: "
        f"{text_vectorizer.get_config()['vocabulary_size']}"
    )

    print(
        f"Output sequence length: "
        f"{text_vectorizer.get_config()['output_sequence_length']}"
    )

    save_text_vec_config(text_vectorizer)



if __name__ == "__main__":
    main()
