import keras

from image_captioning.evaluation.bleu import evaluate_bleu
from image_captioning.model import ShowAttendAndTell


def evaluate_model(
    model: ShowAttendAndTell,
    test_split: dict[str, list[str]],
    feature_maps: dict,
    text_vectorization: keras.layers.TextVectorization,
    max_caption_length: int,
    number_of_examples: int,
) -> dict[str, float]:

    scores = evaluate_bleu(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        text_vectorization=text_vectorization,
        max_caption_length=max_caption_length,
    )

    print_bleu_scores(scores=scores)

    print_generated_captions(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        text_vectorization=text_vectorization,
        max_caption_length=max_caption_length,
        number_of_examples=number_of_examples,
    )

    return scores


def print_bleu_scores(
    scores: dict[str, float],
) -> None:

    print("\n===== BLEU SCORES =====")

    for metric, score in scores.items():

        print(f"{metric}: {score:.4f}")


def print_generated_captions(
    model: ShowAttendAndTell,
    test_split: dict[str, list[str]],
    feature_maps: dict,
    text_vectorization: keras.layers.TextVectorization,
    max_caption_length: int,
    number_of_examples: int,
) -> None:

    print("\n===== GENERATED CAPTIONS =====")

    image_ids = list(test_split)

    for image_id in image_ids[:number_of_examples]:

        if image_id not in feature_maps:

            raise KeyError(
                f'Image ID "{image_id}" is missing '
                "from the feature maps."
            )

        generated_caption = model.generate_caption(
            feature_map=feature_maps[image_id],
            vectorizer=text_vectorization,
            max_caption_length=max_caption_length
        )


        print(f"\nImage: {image_id}")

        print(
            "Generated:",
            " ".join(generated_caption),
        )

        print("References:")

        for caption in test_split[image_id]:

            print(f"- {caption}")
