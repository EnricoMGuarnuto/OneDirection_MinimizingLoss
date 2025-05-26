import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.clip_model.encode_image(x).float()
        return self.head(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_dataset = datasets.ImageFolder(cfg['data']['train_dir'], transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True)

    model = CLIPClassifier(clip_model, embed_dim=512, num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))

    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['num_epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"✅ Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

    torch.save(model.state_dict(), cfg['training']['save_checkpoint'])
    print(f"✅ Model saved to {cfg['training']['save_checkpoint']}")

if __name__ == '__main__':
    main()

