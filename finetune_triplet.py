import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import timm
import open_clip
from triplet_dataset import TripletDataset  # Assicurati di avere questo file

import torchvision.models as tv_models

def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')
    num_classes = cfg['model'].get('num_classes', 1000)

    if name == 'moco_resnet50':
        # Special handling for MoCo v2
        model = tv_models.resnet50(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('module.encoder_q')}
        model.load_state_dict(new_state_dict, strict=False)
        model.fc = nn.Identity()  # remove final classification head
        print(f"✅ Loaded MoCo v2 ResNet50 from {checkpoint_path}")

    elif source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual  # only visual encoder
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    elif source == 'timm':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('classifier.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded custom backbone weights from {checkpoint_path}")
        model.reset_classifier(0, '')  # remove classifier head
        print(f"✅ Loaded {name} from timm")

    else:
        model_fn = getattr(tv_models, name)
        model = model_fn(pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('classifier.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded custom backbone weights from {checkpoint_path}")
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

    # Assicuriamoci che il transform venga passato bene
    dataset = TripletDataset(root_dir=cfg['data']['train_dir'], transform=transform)


    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device)

    optimizer_name = cfg['training'].get('optimizer', 'adam').lower()
    lr = float(cfg['training']['lr_backbone'])  # uso solo uno, perché non abbiamo testa in triplet
    margin = float(cfg['training'].get('margin', 1.0))

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)


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
