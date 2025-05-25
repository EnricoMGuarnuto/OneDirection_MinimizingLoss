import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from train import test_step
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    train_loss, train_acc = 0, 0
    model.train()
    model.to(device)

    for i, batch in enumerate(data_loader):
        
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X, y = batch
        else:
            raise ValueError(f"Batch ha formato inatteso: {type(batch)}, contenuto: {batch}")

        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path al file di configurazione YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n📦 Caricamento dei dataloader")
    if "val_init_args" in cfg["dataset"]:
        train_loader = create_dataloader(cfg, mode='train')
        if isinstance(train_loader, torch.utils.data.DataLoader):
            pass
        elif isinstance(train_loader, (list, tuple)) and isinstance(train_loader[0], torch.utils.data.DataLoader):
            train_loader = train_loader[0]

        val_loader = create_dataloader(cfg, mode='val')
        print("✅ Uso val_loader separato definito da val_init_args")
    else:
        train_loader, val_loader = create_dataloader(cfg, mode='train')
        print("⚠️  Nessun val_loader specificato: faccio split interno dal train")

    from models import get_model
    model = get_model(cfg, mode="classification").to(device)

    print("\n🔍 Verifica dei layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Layer sbloccato: {name}")
        else:
            print(f"❌ Layer bloccato: {name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["training"].get("head_lr", 1e-4))
    num_epochs = cfg["training"].get("num_epochs_head", 5)
    save_path = cfg["training"]["save_name"]

    print("\n🚀 Fase 1: Fine-tune solo testa")
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"⏳ Inizio epoca {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = test_step(model, val_loader, criterion, accuracy_fn, device)
        print(f"[Head Epoch {epoch+1}/{num_epochs}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, save_path)
            print(f"✅ Miglior modello salvato (head): {save_path}")

    print("\n🔓 Sblocco di tutti i layer per full fine-tuning")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"].get("full_lr", 1e-5))
    num_epochs = cfg["training"].get("num_epochs_full", 10)
    best_acc = 0.0

    print("\n🚀 Fase 2: Full fine-tuning")
    for epoch in range(num_epochs):
        print(f"⏳ Inizio epoca {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = test_step(model, val_loader, criterion, accuracy_fn, device)
        print(f"[Full Epoch {epoch+1}/{num_epochs}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, save_path)
            print(f"✅ Miglior modello salvato (full): {save_path}")

    print("\n🏁 Fine training")

if __name__ == "__main__":
    main()
