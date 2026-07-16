from image_captioning.config import SPLITS_FILE, VECTORIZER_CONFIG_FILE
from image_captioning.datasets.tfrecord import create_tfrecord_datasets
from image_captioning.utils.loading import load_splits, load_vectorizer


def main():
    splits = load_splits(SPLITS_FILE)
    text_vectorization = load_vectorizer(VECTORIZER_CONFIG_FILE)
    create_tfrecord_datasets(splits, text_vectorization)


if __name__ == "__main__":
    main()