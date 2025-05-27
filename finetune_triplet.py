import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm
import open_clip
from triplet_dataset import TripletDataset  # salva la classe che ti ho dato prima in questo file

def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')

    if source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual
    else:
        model = timm.create_model(name, pretrained=pretrained, num_classes=0)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"✅ Loaded weights from {checkpoint_path}")

    return model.to(device)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['normalization']['mean'], std=cfg['data']['normalization']['std'])
    ])

    dataset = TripletDataset(cfg['data']['train_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr_backbone']))
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    os.makedirs(os.path.dirname(cfg['training']['save_checkpoint']), exist_ok=True)

    for epoch in range(cfg['training']['num_epochs']):
        print(f"\n🌟 Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), cfg['training']['save_checkpoint'])
        print(f"✅ Saved model to {cfg['training']['save_checkpoint']}")

    print("🏁 Training completed.")

if __name__ == '__main__':
    main()
