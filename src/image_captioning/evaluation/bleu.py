from nltk.translate.bleu_score import corpus_bleu


def evaluate_bleu(
    captions: dict[str, dict[str, str | list[str]]],
) -> dict[str, float]:

    references = []
    hypotheses = []

    for data in captions.values():

        refs = [caption.split() for caption in data["reference"]]
        hyp = data["generated"].split()

        references.append(refs)
        hypotheses.append(hyp)

    return {
        "BLEU-1": corpus_bleu(
            references,
            hypotheses,
            weights=(1.0, 0.0, 0.0, 0.0),
        ),
        "BLEU-2": corpus_bleu(
            references,
            hypotheses,
            weights=(0.5, 0.5, 0.0, 0.0),
        ),
        "BLEU-3": corpus_bleu(
            references,
            hypotheses,
            weights=(1/3, 1/3, 1/3, 0.0),
        ),
        "BLEU-4": corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
        ),
    }