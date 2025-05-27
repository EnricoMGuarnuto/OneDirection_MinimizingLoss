import optuna
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from finetune import load_model  # usa il tuo script finetune.py

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

def objective(trial):
    with open('config/resnet50v1.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr_head = trial.suggest_float('lr_head', 1e-4, 1e-2, log=True)
    lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 1e-4, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    freeze_backbone = trial.suggest_categorical('freeze_backbone', [False, True])

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
    model = load_model(cfg, device, num_classes=num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        print("✅ Backbone frozen")

    params_to_optimize = []
    head_params = [p for n, p in model.named_parameters() if ('fc' in n or 'classifier' in n) and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and ('fc' not in n and 'classifier' not in n)]

    if head_params:
        params_to_optimize.append({'params': head_params, 'lr': lr_head})
    if backbone_params:
        params_to_optimize.append({'params': backbone_params, 'lr': lr_backbone})

    if not params_to_optimize:
        raise ValueError("⚠ Nessun parametro da ottimizzare! Controlla freeze_backbone o aggiungi un head.")

    if optimizer_name == 'adam':
        optimizer = optim.Adam(params_to_optimize)
    else:
        optimizer = optim.SGD(params_to_optimize, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    num_epochs = cfg['training']['num_epochs']
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        model.eval()
        val_loss, val_acc = train_one_epoch(model, val_loader, optimizer, loss_fn, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return -best_val_acc  # minimize negative accuracy → maximize accuracy

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("✅ Best hyperparameters:", study.best_params)
