import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np

class HockeyDataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None, max_frames=20):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.transform = transform
        self.max_frames = max_frames  # limit frames to save memory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['filepath']
        label = int(self.data.iloc[idx]['label'])

        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count >= self.max_frames:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # normalize
            frames.append(frame)
            count += 1
        cap.release()

        # If less frames than max_frames, pad with zeros
        while len(frames) < self.max_frames:
            frames.append(np.zeros((112,112,3)))

        frames = np.array(frames, dtype=np.float32)
        frames = torch.tensor(frames).permute(3,0,1,2)  # (C,T,H,W)
        label = torch.tensor(label, dtype=torch.long)
        return frames, label
