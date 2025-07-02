import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinearHead(nn.Module):
    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def load_model(cfg, device, num_classes):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')

    if source == 'torchvision':
        model_fn = getattr(models, name)
        model = model_fn(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print(f"Loaded weights from {checkpoint_path}")
        else:
            print(f"Loaded {name} with pretrained={pretrained}")

    return model.to(device)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


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
    num_classes = len(full_dataset.classes)
    train_loader = DataLoader(full_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device, num_classes)

    if cfg['training'].get('freeze_backbone', False):
        for name, param in model.named_parameters():
            if 'head' not in name and 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        print("Backbone frozen")

    lr_head = float(cfg['training']['lr_head'])
    lr_backbone = float(cfg['training']['lr_backbone'])

    head_params = [p for n, p in model.named_parameters() if ('head' in n or 'fc' in n or 'classifier' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and ('head' not in n and 'fc' not in n and 'classifier' not in n)]

    optimizer = optim.Adam([
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone}
    ])
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(cfg['training']['save_checkpoint']), exist_ok=True)

    save_path = cfg['training']['save_checkpoint']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_loss = float('inf')

    for epoch in range(cfg['training']['num_epochs']):
        print(f"\n Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (loss improved)")
        else:
            print("Loss did not improve. Skipping save.")


        print("Final training completed. Model ready for retrieval!")


if __name__ == '__main__':
    main()
