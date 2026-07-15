import json

from keras.layers import TextVectorization

from image_captioning.config.paths import SPLIT_FILE, TEXT_VECTORIZATION_CONFIG_FILE



def load_split_json() -> dict[str, dict[str, list[str]]]:
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_training_captions() -> list[str]:
    split_json = load_split_json()
    train_split = split_json["train"]
    training_captions = []
    for _, image_captions in train_split.items():
        training_captions.extend(image_captions)
    return training_captions

def load_text_vectorization_config():
    with open(TEXT_VECTORIZATION_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text_vectorization() -> TextVectorization:
    return TextVectorization.from_config(
        load_text_vectorization_config()
    )