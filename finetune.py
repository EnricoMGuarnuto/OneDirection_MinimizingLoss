import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import timm
import open_clip


def load_model(cfg, device, num_classes):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')

    if source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual  # solo parte visiva per fine-tuning
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    elif source == 'timm':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Loaded weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained={pretrained}")

    else:
        model_fn = getattr(models, name)
        model = model_fn(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Loaded weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained={pretrained}")

    return model.to(device)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

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

    full_dataset = datasets.ImageFolder(cfg['data']['train_dir'], transform=transform)
    val_size = int(len(full_dataset) * cfg['data'].get('val_split', 0.2))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=4)

    num_classes = len(full_dataset.classes)
    model = load_model(cfg, device, num_classes)

    if cfg['training'].get('freeze_backbone', False):
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        print("✅ Backbone frozen")

    lr_head = float(cfg['training']['lr_head'])
    lr_backbone = float(cfg['training']['lr_backbone'])

    params_to_optimize = []
    head_params = [p for n, p in model.named_parameters() if ('fc' in n or 'classifier' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and ('fc' not in n and 'classifier' not in n)]
    
    if head_params:
        params_to_optimize.append({'params': head_params, 'lr': lr_head})

    if backbone_params:
        params_to_optimize.append({'params': backbone_params, 'lr': lr_backbone})

    optimizer = optim.Adam(params_to_optimize)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(cfg['training']['num_epochs']):
        print(f"\n🌟 Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(cfg['training']['save_checkpoint']), exist_ok=True)
            torch.save(model.state_dict(), cfg['training']['save_checkpoint'])
            print(f"✅ Saved best model to {cfg['training']['save_checkpoint']}")

    print("🏁 Training completed.")

if __name__ == '__main__':
    main()