import os
import csv

base_dir = "datasets/hockey"
splits = ["train", "val"]  # only existing splits
categories = {"fight": 1, "nonfight": 0}

csv_file = "hockey_dataset.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "split"])
    
    for split in splits:
        for cat, label in categories.items():
            folder = os.path.join(base_dir, split, cat)
            if not os.path.exists(folder):
                print(f"Warning: Folder not found {folder}")
                continue
            for file in os.listdir(folder):
                if file.endswith(".mp4"):
                    path = os.path.join(folder, file)
                    writer.writerow([path, label, split])

print(f"CSV file '{csv_file}' created successfully!")
