import os
import shutil
import random

# ===== PATHS =====

SOURCE_DIR = "/Users/belhajali/Desktop/asl_top100_dataset"
OUTPUT_DIR = "data/asl_split"


TRAIN_COUNT = 21
VAL_COUNT = 5
TEST_COUNT = 5


# ===== CREATE SPLIT FOLDERS =====

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)


# ===== SPLIT DATASET =====

for word in os.listdir(SOURCE_DIR):

    word_path = os.path.join(SOURCE_DIR, word)

    if not os.path.isdir(word_path):
        continue

    videos = [v for v in os.listdir(word_path) if v.endswith(".mp4")]

    if len(videos) < 31:
        print(f"Skipping {word}, not enough videos")
        continue

    random.shuffle(videos)

    train_videos = videos[:TRAIN_COUNT]
    val_videos = videos[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]
    test_videos = videos[TRAIN_COUNT+VAL_COUNT:TRAIN_COUNT+VAL_COUNT+TEST_COUNT]


    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos
    }


    for split_name, split_files in splits.items():

        split_word_dir = os.path.join(OUTPUT_DIR, split_name, word)

        os.makedirs(split_word_dir, exist_ok=True)

        for file in split_files:

            src = os.path.join(word_path, file)
            dst = os.path.join(split_word_dir, file)

            shutil.copy(src, dst)


print("Dataset split completed successfully")