from nltk.translate.bleu_score import corpus_bleu


def evaluate_bleu(
    predictions: dict[str, str],
    references: dict[str, list[str]],
) -> dict[str, float]:

    bleu_references = []
    hypotheses = []

    for image_id, generated in predictions.items():

        if image_id not in references:
            raise KeyError(
                f'Image ID "{image_id}" is missing from the references.'
            )

        bleu_references.append(
            [caption.split() for caption in references[image_id]]
        )

        hypotheses.append(
            generated.split()
        )

    return {
        "BLEU-1": corpus_bleu(
            bleu_references,
            hypotheses,
            weights=(1.0, 0.0, 0.0, 0.0),
        ),
        "BLEU-2": corpus_bleu(
            bleu_references,
            hypotheses,
            weights=(0.5, 0.5, 0.0, 0.0),
        ),
        "BLEU-3": corpus_bleu(
            bleu_references,
            hypotheses,
            weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
        ),
        "BLEU-4": corpus_bleu(
            bleu_references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
        ),
    }