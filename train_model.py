import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# Dataset
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, csv_file, split='train', max_frames=40):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row['filepath']
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # If video has fewer frames, pad with last frame
        while len(frames) < self.max_frames:
            frames.append(frames[-1])

        frames = np.array(frames)  # shape: [num_frames, H, W, 3]
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        # shape: [3, num_frames, 64, 64]

        return frames, label

# ----------------------------
# Model
# ----------------------------
class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 40 * 16 * 16, 64),  # 40 frames, 16x16 spatial after pooling
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------------
# Training
# ----------------------------
def train_model(csv_file, epochs=5, batch_size=1, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VideoDataset(csv_file, split='train')
    val_dataset = VideoDataset(csv_file, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Conv3DModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            frames = frames.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(frames)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "violence_model.pth")
    print("Model saved as 'violence_model.pth'")

# ----------------------------
# Run training
# ----------------------------
if __name__ == "__main__":
    train_model("hockey_dataset.csv", epochs=5)
