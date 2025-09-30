import torch
import torch.nn as nn
import cv2
import numpy as np
from train_model import Conv3DModel

def predict(video_path, model_path="violence_model.pth", max_frames=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Conv3DModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Pad frames if fewer than max_frames
    while len(frames) < max_frames and len(frames) > 0:
        frames.append(frames[-1])

    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    frames = frames.to(device)

    with torch.no_grad():
        output = model(frames)
        prediction = (output > 0.5).float().item()
    
    if prediction == 1:
        return "Fight"
    else:
        return "Non-fight"


# Example usage:
if __name__ == "__main__":
    video_file = "datasets/hockey/test/video1.mp4"  # replace with your video path
    print("Prediction:", predict(video_file))
