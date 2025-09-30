import os
import cv2

# Set the folders
folders = [
    "datasets/hockey/train/fight_mp4",
    "datasets/hockey/train/nonfight_mp4",
    "datasets/hockey/val/fight_mp4",
    "datasets/hockey/val/nonfight_mp4"
]

# Number of frames your model expects
MIN_FRAMES = 16

# Option: set REMOVE = True to delete short videos, False to just list them
REMOVE = True

for folder in folders:
    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):
        if not file.endswith(".mp4"):
            continue
        path = os.path.join(folder, file)
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count < MIN_FRAMES:
            if REMOVE:
                os.remove(path)
                print(f"Deleted short video: {path} ({frame_count} frames)")
            else:
                print(f"Too short: {path} ({frame_count} frames)")
