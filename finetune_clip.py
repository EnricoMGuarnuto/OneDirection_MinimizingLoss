import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import clip
import random

class TripletImageDataset(Dataset):
    def __init__(self, root, preprocess):
        self.dataset = datasets.ImageFolder(root, transform=preprocess)
        self.class_to_indices = self._build_index()

    def _build_index(self):
        index = {}
        for idx, (_, label) in enumerate(self.dataset):
            index.setdefault(label, []).append(idx)
        return index

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]
        while True:
            positive_index = random.choice(self.class_to_indices[anchor_label])
            if positive_index != index:
                break
        positive_img, _ = self.dataset[positive_index]
        while True:
            negative_label = random.choice(list(self.class_to_indices.keys()))
            if negative_label != anchor_label:
                break
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]
        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset)

def normalize(tensor):
    return nn.functional.normalize(tensor, p=2, dim=1)

class CLIPWithHead(nn.Module):
    def __init__(self, clip_model, output_dim=512):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.head = nn.Linear(512, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.clip.encode_image(x).to(torch.float32)
        return normalize(self.head(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    model = CLIPWithHead(clip_model, output_dim=cfg['model']['output_dim']).to(device)

    dataset = TripletImageDataset(cfg['data']['train_dir'], preprocess)
    loader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.head.parameters(), lr=float(cfg['training']['lr']))
    criterion = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        total_loss = 0

        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch+1}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_feat = model(anchor)
            positive_feat = model(positive)
            negative_feat = model(negative)

            loss = criterion(anchor_feat, positive_feat, negative_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} - Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), cfg['training']['save_checkpoint'])
    print(f"✅ Saved fine-tuned CLIP head to {cfg['training']['save_checkpoint']}")

if __name__ == '__main__':
    main()
