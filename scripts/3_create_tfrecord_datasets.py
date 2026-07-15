from image_captioning.datasets.tfrecord import create_tfrecord_datasets
from image_captioning.utils.loading import load_split_json, load_text_vectorization


def main():
    split = load_split_json()
    text_vectorization = load_text_vectorization()
    create_tfrecord_datasets(split, text_vectorization)


if __name__ == "__main__":
    main()