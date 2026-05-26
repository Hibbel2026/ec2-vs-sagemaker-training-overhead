import os
from collections import defaultdict

DATASET_DIR = "/Users/belhajali/Desktop/asl_dataset/American-Sign-Language-Dataset"

word_count = defaultdict(int)

# loop through all parts
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


print("Total words:", len(word_count))
print()

# sort words by number of videos
sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

print("Top 100 words with most videos:\n")

for word, count in sorted_words[:100]:
    print(f"{word:20} {count}")