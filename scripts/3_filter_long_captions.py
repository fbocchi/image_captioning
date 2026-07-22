import json

from config import SPLITS_FILE, NEW_SPLITS_FILE, MAX_CAPTION_LEN

from image_captioning.utils import load_splits


def drop_longer_captions(
    splits: dict[str, dict[str, list[str]]],
    max_length: int,
) -> dict[str, dict[str, list[str]]]:

    new_splits = {}

    total_captions = 0
    removed_captions = 0

    for split_name, split in splits.items():

        new_splits[split_name] = {}

        for image_name, image_captions in split.items():

            filtered_captions = []

            for caption in image_captions:

                total_captions += 1

                if len(caption.split()) <= max_length:
                    filtered_captions.append(caption)
                else:
                    removed_captions += 1

            if filtered_captions:
                new_splits[split_name][image_name] = filtered_captions

    kept_captions = total_captions - removed_captions

    print(f"Maximum caption length: {max_length}")
    print(f"Total captions: {total_captions}")
    print(f"Kept captions: {kept_captions}")
    print(f"Removed captions: {removed_captions}")
    print(f"Removed percentage: {100 * removed_captions / total_captions:.2f}%")

    return new_splits


def main():
    splits = load_splits(SPLITS_FILE)
    new_splits = drop_longer_captions(splits, MAX_CAPTION_LEN)
    with open(NEW_SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_splits, f, indent=4)


if __name__ == "__main__":
    main()