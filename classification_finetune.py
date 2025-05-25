import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from train import train_step, test_step
import os


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path al file di configurazione YAML')
    args = parser.parse_args()

    # Carica configurazione
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica train loader
    train_loader, val_loader = None, None

    if "val_init_args" in cfg["dataset"]:
        # Se specificato, usa val loader separato
        train_loader, _, _, _ = create_dataloader(
            root_dir=cfg["dataset"]["path_root"],
            batch_size=cfg["data"]["batch_size_train"],
            val_split=0,
            mode='train',
            dataset_name=cfg["dataset"]["name"],
            init_args=cfg["dataset"].get("train_init_args", cfg["dataset"].get("init_args", {}))
        )

        val_loader, _, _, _ = create_dataloader(
            root_dir=cfg["dataset"]["path_root"],
            batch_size=cfg["data"]["batch_size_test"],
            val_split=0,
            mode='test',
            dataset_name=cfg["dataset"]["name"],
            init_args=cfg["dataset"].get("val_init_args", cfg["dataset"].get("init_args", {}))
        )
        print("✅ Uso val_loader separato definito da val_init_args")

    else:
        # Se no, fai split interno
        train_loader, val_loader, _, _ = create_dataloader(
            root_dir=cfg["dataset"]["path_root"],
            batch_size=cfg["data"]["batch_size_train"],
            val_split=cfg["data"].get("val_split", 0.2),
            mode='train',
            dataset_name=cfg["dataset"]["name"],
            init_args=cfg["dataset"].get("init_args", {})
        )
        print("⚠️  Nessun val_loader specificato: faccio split interno dal train")

    # Carica modello
    from models import get_model
    model = get_model(cfg, mode="classification")
    model = model.to(device)

    print("\n🔍 Verifica dei layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Layer sbloccato: {name}")
        else:
            print(f"❌ Layer bloccato: {name}")

    # Fase 1: ottimizza solo la testa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["training"].get("head_lr", 1e-4))
    num_epochs = cfg["training"].get("num_epochs_head", 5)
    save_path = cfg["training"]["save_name"]

    print("\n🚀 Fase 1: Fine-tune solo testa")
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = test_step(model, val_loader, criterion, accuracy_fn, device)
        print(f"[Head Epoch {epoch+1}/{num_epochs}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, save_path)
            print(f"✅ Miglior modello salvato (head): {save_path}")

    # Fase 2: sblocca tutto e ottimizza
    print("\n🔓 Sblocco di tutti i layer per full fine-tuning")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"].get("full_lr", 1e-5))
    num_epochs = cfg["training"].get("num_epochs_full", 10)
    best_acc = 0.0

    print("\n🚀 Fase 2: Full fine-tuning")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, accuracy_fn, device)
        val_loss, val_acc = test_step(model, val_loader, criterion, accuracy_fn, device)
        print(f"[Full Epoch {epoch+1}/{num_epochs}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, save_path)
            print(f"✅ Miglior modello salvato (full): {save_path}")

    print("\n🏁 Fine training")

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

if __name__ == "__main__":
    main()
