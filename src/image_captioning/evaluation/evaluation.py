import keras

from image_captioning.evaluation.bleu import evaluate_bleu
from image_captioning.model import ShowAttendAndTell


def evaluate_model(
    model: ShowAttendAndTell,
    test_split: dict[str, list[str]],
    feature_maps: dict,
    text_vectorization: keras.layers.TextVectorization,
    max_caption_length: int,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, str | list[str]]]
]:

    captions = generate_captions(
        model=model,
        test_split=test_split,
        feature_maps=feature_maps,
        text_vectorization=text_vectorization,
        max_caption_length=max_caption_length,
    )

    scores = evaluate_bleu(captions)

    return scores, captions


def generate_captions(
    model: ShowAttendAndTell,
    test_split: dict[str, list[str]],
    feature_maps: dict,
    text_vectorization: keras.layers.TextVectorization,
    max_caption_length: int,
) -> dict[str, dict[str, str | list[str]]]:

    captions = {}

    for image_id, references in test_split.items():

        if image_id not in feature_maps:
            raise KeyError(
                f'Image ID "{image_id}" is missing from the feature maps.'
            )

        generated = model.generate_caption(
            feature_map=feature_maps[image_id],
            vectorizer=text_vectorization,
            max_caption_length=max_caption_length,
        )

        captions[image_id] = {
            "reference": references,
            "generated": " ".join(generated),
        }

    return captions