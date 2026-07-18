from image_captioning.evaluation.bleu import evaluate_bleu


def evaluate_test_captions(
    predictions: dict[str, str],
    test_split: dict[str, list[str]],
) -> dict[str, float]:

    captions = {}

    for image_id, generated in predictions.items():

        if image_id not in test_split:
            raise KeyError(
                f'Image ID "{image_id}" is missing from the test split.'
            )

        captions[image_id] = {
            "reference": test_split[image_id],
            "generated": generated,
        }

    return evaluate_bleu(captions)