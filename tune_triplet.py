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

    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    margin = trial.suggest_float('margin', 0.2, 1.0)
    freeze_backbone = trial.suggest_categorical('freeze_backbone', [False, True])

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['data']['normalization']['mean'], std=cfg['data']['normalization']['std'])
    ])

    dataset = TripletDataset(cfg['data']['train_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    model = load_model(cfg, device)

    has_head = any(name.startswith('fc') or name.startswith('classifier') for name, _ in model.named_parameters())

    if freeze_backbone and not has_head:
        print("⚠ Modello senza testa, forzo freeze_backbone=False")
        freeze_backbone = False

    params_to_optimize = []
    if not freeze_backbone:
        params_to_optimize += [p for p in model.parameters() if p.requires_grad]

    if not params_to_optimize:
        raise ValueError("⚠ Nessun parametro da ottimizzare! Controlla freeze_backbone o aggiungi un head.")

    if optimizer_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=lr)
    else:
        optimizer = optim.SGD(params_to_optimize, lr=lr, momentum=0.9)

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
    return epoch_loss



    params_to_optimize = []
    if not freeze_backbone:
        params_to_optimize += [p for p in model.parameters() if p.requires_grad]

    if not params_to_optimize:
        raise ValueError("⚠ Nessun parametro da ottimizzare! Controlla freeze_backbone o aggiungi un head.")

    if optimizer_name == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=lr_head)
    else:
        optimizer = optim.SGD(params_to_optimize, lr=lr_head, momentum=0.9)

    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
    return epoch_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("✅ Best hyperparameters:", study.best_params)
