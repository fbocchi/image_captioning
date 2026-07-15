import keras
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from image_captioning.model import ShowAttendAndTell


def evaluate_bleu(
    model: ShowAttendAndTell,
    test_split: dict[str, list[str]],
    feature_maps: dict,
    text_vectorization: keras.layers.TextVectorization,
    max_caption_length: int,
) -> dict[str, float]:

    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []

    for image_id, captions in tqdm(
        test_split.items(),
        total=len(test_split),
        desc="Evaluating BLEU",
        unit="image",
    ):
        if image_id not in feature_maps:

            raise KeyError(
                f'Image ID "{image_id}" is missing '
                "from the feature maps."
            )

        generated_caption = model.generate_caption(
            feature_map=feature_maps[image_id],
            vectorizer=text_vectorization,
            max_caption_length=max_caption_length,
        )

        image_references = [
            caption.split()
            for caption in captions
        ]

        references.append(image_references)
        hypotheses.append(generated_caption)

    return {
        "BLEU-1": corpus_bleu(
            references,
            hypotheses,
            weights=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        ),
        "BLEU-2": corpus_bleu(
            references,
            hypotheses,
            weights=(
                0.5,
                0.5,
                0.0,
                0.0,
            ),
        ),
        "BLEU-3": corpus_bleu(
            references,
            hypotheses,
            weights=(
                1 / 3,
                1 / 3,
                1 / 3,
                0.0,
            ),
        ),
        "BLEU-4": corpus_bleu(
            references,
            hypotheses,
            weights=(
                0.25,
                0.25,
                0.25,
                0.25,
            ),
        ),
    }