# backend/inference.py
import torch
import os
from model_pytorch import FrameRNNModel, Video3DModel
from utils import extract_frames_opencv
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, arch='rnn', arch_3d='r3d_18'):
    if arch == 'rnn':
        model = FrameRNNModel(pretrained=False)
    else:
        model = Video3DModel(arch=arch_3d, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_video(video_path, model, num_frames=16, window_stride=8, sliding=False):
    """
    If sliding==False: sample fixed num_frames and return a single score.
    If sliding==True: computes a sliding-window score for each window and returns max score + timestamps.
    """
    frames = extract_frames_opencv(video_path, num_frames=num_frames, resize=(224,224))
    # frames: (T,H,W,C)
    # convert to torch tensor shaped (1,T,C,H,W)
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensors = [transform(f) for f in frames]
    seq = torch.stack(tensors, dim=0)  # (T,C,H,W)
    seq = seq.unsqueeze(0)  # (1,T,C,H,W)
    seq = seq.to(DEVICE)
    with torch.no_grad():
        score = model(seq)  # (1,) or (1,N) for 3D? our models return (B,)
        if isinstance(score, torch.Tensor):
            s = float(score.cpu().item())
        else:
            s = float(score)
    return {'violence_score': s, 'label': 'violent' if s >= 0.5 else 'non-violent'}
