import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from finetune_triplet import load_model
from triplet_dataset import TripletDataset
import os
import argparse

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for anchor, positive, negative in dataloader:
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

def objective(trial, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    margin = trial.suggest_float('margin', 0.1, 1.0)

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['normalization']['mean'], std=cfg['data']['normalization']['std'])
    ])

    dataset = TripletDataset(cfg['data']['train_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    num_epochs = cfg['training'].get('num_epochs', 10)
    best_loss = float('inf')

    os.makedirs('best_models', exist_ok=True)
    checkpoint_path = f"best_models/model_triplet_trial_{trial.number}.pt"

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)

    return best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, cfg), n_trials=30)
    print("✅ Best hyperparameters:", study.best_params)
