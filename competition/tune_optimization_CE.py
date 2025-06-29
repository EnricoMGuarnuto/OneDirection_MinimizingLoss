import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from finetune import load_model
import os
import argparse

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def objective(trial, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr_head = trial.suggest_float('lr_head', 1e-5, 1e-2, log=True)
    lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['normalization']['mean'], std=cfg['data']['normalization']['std'])
    ])

    full_dataset = datasets.ImageFolder(cfg['data']['train_dir'], transform=transform)
    num_classes = len(full_dataset.classes)

    val_size = int(len(full_dataset) * cfg['data'].get('val_split', 0.2))
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=4)

    model = load_model(cfg, device, num_classes)

    head_params = [p for n, p in model.named_parameters() if ('fc' in n or 'classifier' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and ('fc' not in n and 'classifier' not in n)]

    params_to_optimize = []
    if head_params:
        params_to_optimize.append({'params': head_params, 'lr': lr_head})
    if backbone_params:
        params_to_optimize.append({'params': backbone_params, 'lr': lr_backbone})

    if optimizer_name == 'adam':
        optimizer = optim.Adam(params_to_optimize)
    else:
        optimizer = optim.SGD(params_to_optimize, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    num_epochs = cfg['training']['num_epochs']

    os.makedirs('best_models', exist_ok=True)
    checkpoint_path = f"best_models/model_ce_trial_{trial.number}.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    return -best_val_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, cfg), n_trials=30)
    print("✅ Best hyperparameters:", study.best_params)
