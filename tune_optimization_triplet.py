import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from finetune_triplet import load_model
from triplet_dataset import TripletDataset

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

def objective(trial):
    with open('config/moco_resnet50.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

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

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    num_epochs = cfg['training'].get('num_epochs', 5)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    return best_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("✅ Best hyperparameters:", study.best_params)