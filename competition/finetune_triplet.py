import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models as tv_models
from tqdm import tqdm
import timm
import open_clip
from torch.utils.data import Dataset
from PIL import Image
import random

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Solo directory (classi), esclude .DS_Store e file simili
        self.classes = [d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d))]

        # Mappa classe -> lista di immagini
        self.class_to_imgs = {}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            images = [os.path.join(cls_path, img)
                      for img in os.listdir(cls_path)
                      if img.lower().endswith(('.png', '.jpg', '.jpeg')) and
                         os.path.isfile(os.path.join(cls_path, img))]
            if len(images) >= 2:
                self.class_to_imgs[cls] = images

        self.triplet_sources = [(cls, img_path)
                                for cls, imgs in self.class_to_imgs.items()
                                for img_path in imgs]

    def __len__(self):
        return len(self.triplet_sources)

    def __getitem__(self, idx):
        anchor_cls, anchor_path = self.triplet_sources[idx]
        positive_path = random.choice([p for p in self.class_to_imgs[anchor_cls] if p != anchor_path])
        negative_cls = random.choice([c for c in self.classes if c != anchor_cls])
        negative_path = random.choice(self.class_to_imgs[negative_cls])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')

    if name == 'moco_resnet50':
        model = tv_models.resnet50(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('module.encoder_q')}
        model.load_state_dict(new_state_dict, strict=False)
        model.fc = nn.Identity()
        print(f"✅ Loaded MoCo v2 ResNet50 from {checkpoint_path}")

    elif source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    elif source == 'timm':
        model = timm.create_model(name, pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights from {checkpoint_path}")
        model.reset_classifier(0)
        print(f"✅ Loaded {name} from timm")

    else:
        model_fn = getattr(tv_models, name)
        model = model_fn(pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights from {checkpoint_path}")
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        print(f"✅ Loaded {name} from torchvision")

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
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

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
