# backend/model_pytorch.py
import torch
import torch.nn as nn
import torchvision.models as models

class FrameRNNModel(nn.Module):
    """
    Per-frame CNN (MobileNetV2) -> BiLSTM -> MLP head
    Input: (B, T, C, H, W)
    Output: (B,) scores between 0..1
    """
    def __init__(self, feat_dim=1280, lstm_hidden=256, lstm_layers=1, bidirectional=True, pretrained=True):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        # Use feature extractor up to last conv (features)
        self.feature_extractor = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.feat_dim = feat_dim  # MobileNetV2 last channel
        self.lstm = nn.LSTM(self.feat_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)  # (B*T, feat, h, w)
        feats = self.avgpool(feats).view(B, T, -1)  # (B,T,feat_dim)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.classifier(last).squeeze(1)

class Video3DModel(nn.Module):
    """
    Wrapper around torchvision video models (r3d_18 / mc3_18 / r2plus1d_18)
    Expects input shape (B, C, T, H, W) as torchvision video models expect channel-first with time dim after channels.
    We'll accept (B,T,C,H,W) and permute.
    """
    def __init__(self, arch='r3d_18', pretrained=False):
        super().__init__()
        if arch == 'r3d_18':
            self.net = models.video.r3d_18(weights=(models.VideoResNet_R3D_18_Weights.KINETICS400_V1 if pretrained else None))
            in_features = self.net.fc.in_features
            # replace final fc
            self.net.fc = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )
        elif arch == 'mc3_18':
            self.net = models.video.mc3_18(weights=(models.VideoResNet_MC3_18_Weights.KINETICS400_V1 if pretrained else None))
            in_features = self.net.fc.in_features
            self.net.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        else:
            raise ValueError("Unsupported arch")
    def forward(self, x):
        # x: (B, T, C, H, W) -> convert to (B, C, T, H, W)
        x = x.permute(0,2,1,3,4)
        return self.net(x).squeeze(1)
