import os
import shutil
from collections import defaultdict

# ----- PATHS -----

DATASET_DIR = "/Users/belhajali/Desktop/asl_dataset/American-Sign-Language-Dataset"
OUTPUT_DIR = "/Users/belhajali/Desktop/asl_top100_dataset"


# ----- COUNT VIDEOS -----

word_count = defaultdict(int)

for part in os.listdir(DATASET_DIR):

    part_path = os.path.join(DATASET_DIR, part)

    if not os.path.isdir(part_path):
        continue

    for file in os.listdir(part_path):

        if file.endswith(".mp4") and "-" in file:

            try:
                word = file.split("-",1)[1].replace(".mp4","").lower()
                word_count[word] += 1
            except:
                continue


# ----- FIND TOP 100 WORDS -----

sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

top100 = [w for w,_ in sorted_words[:100]]

print("Top 100 words selected")


# ----- CREATE OUTPUT DATASET -----

os.makedirs(OUTPUT_DIR, exist_ok=True)

for word in top100:
    os.makedirs(os.path.join(OUTPUT_DIR, word), exist_ok=True)


# ----- COPY VIDEOS -----

copied = 0

for part in os.listdir(DATASET_DIR):

    part_path = os.path.join(DATASET_DIR, part)

    if not os.path.isdir(part_path):
        continue

    for file in os.listdir(part_path):

        if file.endswith(".mp4") and "-" in file:

            try:
                word = file.split("-",1)[1].replace(".mp4","").lower()

                if word in top100:

                    src = os.path.join(part_path, file)
                    dst = os.path.join(OUTPUT_DIR, word, file)

                    shutil.copy(src, dst)

                    copied += 1

            except:
                continue


print("Finished!")
print("Videos copied:", copied)
print("New dataset location:", OUTPUT_DIR)