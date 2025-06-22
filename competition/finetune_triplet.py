import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import open_clip
from triplet_dataset import TripletDataset
from torchvision import models as tv_models
import timm  # if needed for future support

def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'open_clip')  # default to open_clip for backward compatibility
    checkpoint_path = cfg['model'].get('checkpoint_path', '')
    pretrained = cfg['model'].get('pretrained', True)

    if source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    elif source == 'torchvision':
        model_fn = getattr(tv_models, name)
        model = model_fn(pretrained=pretrained)
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded ResNet50 weights from {checkpoint_path}")
        print(f"✅ Loaded {name} from torchvision")

    else:
        raise ValueError(f"Unsupported model source: {source}")

    return model.to(device)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
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

    dataset = TripletDataset(root_dir=cfg['data']['train_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr_backbone']))
    loss_fn = nn.TripletMarginLoss(margin=cfg['training'].get('margin', 1.0), p=2)

    save_path = cfg['training']['save_checkpoint']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float('inf')

    for epoch in range(cfg['training']['num_epochs']):
        print(f"\n🌟 Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path} (loss improved)")
        else:
            print("ℹ️ Loss did not improve. Skipping save.")

    print("🏁 Final training completed. Model ready for retrieval!")

if __name__ == '__main__':
    main()
