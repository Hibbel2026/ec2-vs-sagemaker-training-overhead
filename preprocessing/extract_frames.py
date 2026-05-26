import os
import cv2
import numpy as np

# ===== SETTINGS =====

VIDEO_DIR = "/Users/belhajali/Desktop/asl_split"
OUTPUT_DIR = "/tmp/local_frames"

SEQUENCE_LENGTH = 16


# ===== FUNCTION TO EXTRACT FRAMES =====

def extract_frames(video_path, output_folder):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return

    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)

    frame_id = 0
    saved_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id in frame_indices:

            filename = os.path.join(output_folder, f"frame_{saved_id:03d}.jpg")
            cv2.imwrite(filename, frame)

            saved_id += 1

        frame_id += 1

    cap.release()


# ===== MAIN LOOP =====

for split in ["train", "val", "test"]:

    split_path = os.path.join(VIDEO_DIR, split)

    for word in os.listdir(split_path):

        word_path = os.path.join(split_path, word)

        if not os.path.isdir(word_path):
            continue

        for video in os.listdir(word_path):

            if not video.endswith(".mp4"):
                continue

            video_path = os.path.join(word_path, video)

            video_name = video.replace(".mp4", "")

            output_folder = os.path.join(
                OUTPUT_DIR,
                split,
                word,
                video_name
            )

            os.makedirs(output_folder, exist_ok=True)

            extract_frames(video_path, output_folder)


print("Frame extraction completed.")