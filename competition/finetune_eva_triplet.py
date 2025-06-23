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

def load_model(cfg, device):
    name = cfg['model']['name']
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained='merged2b_s8b_b131k')
    model = model.visual  # solo visual encoder
    if cfg['model'].get('checkpoint_path'):
        state_dict = torch.load(cfg['model']['checkpoint_path'], map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded custom weights from {cfg['model']['checkpoint_path']}")
    else:
        print(f"✅ Loaded {name} with pretrained weights from open_clip")
    return model.to(device)

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()  # ✅ Mixed precision scaler

    for anchor, positive, negative in tqdm(dataloader, desc="Training"):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # ✅ autocast for mixed precision
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        scaler.scale(loss).backward()   # ✅ scale loss for backward
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # (opzionale) Libera cache a ogni batch per evitare frammentazione
        torch.cuda.empty_cache()

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
            print("ℹ Loss did not improve. Skipping save.")

    print("🏁 Final training completed. Model ready for retrieval!")

if _name_ == '_main_':
    main()