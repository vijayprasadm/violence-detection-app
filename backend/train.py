# backend/train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FightVideoDataset
from model_pytorch import FrameRNNModel, Video3DModel
from sklearn.metrics import roc_auc_score
import time
import os

def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def eval_loop(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            ys.append(labels.numpy())
            preds.append(outputs.cpu().numpy())
    import numpy as np
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    try:
        auc = roc_auc_score(ys, preds)
    except Exception:
        auc = 0.5
    return auc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print("Using device:", device)
    # Load samples: expect CSV or list
    # For simplicity, use a CSV file: each line "path,label"
    def read_csv(p):
        samples=[]
        with open(p,'r') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                path,label=line.split(',')
                samples.append((path.strip(), int(label.strip())))
        return samples
    train_samples = read_csv(args.train_csv)
    val_samples = read_csv(args.val_csv)
    train_ds = FightVideoDataset(train_samples, num_frames=args.num_frames)
    val_ds = FightVideoDataset(val_samples, num_frames=args.num_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    if args.arch == 'rnn':
        model = FrameRNNModel(pretrained=args.pretrained).to(device)
    else:
        model = Video3DModel(arch=args.arch_3d, pretrained=args.pretrained).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_auc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_auc = eval_loop(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_auc={val_auc:.4f} time={(time.time()-t0):.1f}s")
        # checkpoint
        ckpt = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
    print("Training complete. Best AUC:", best_auc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--arch', choices=['rnn','3d'], default='rnn')
    parser.add_argument('--arch_3d', default='r3d_18', help='if arch==3d choose r3d_18 or mc3_18')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--force_cpu', action='store_true', help='Use CPU even if GPU available')
    args = parser.parse_args()
    main(args)
