import torch
from train import train_step, test_step  # Assicurati che train_step e test_step siano definiti in train.py
import torch.nn as nn
import torch.optim as optim
import os

def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

def train_model(model, train_loader, val_loader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    save_path = cfg["training"]["save_name"]
    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        val_loss, val_acc = test_step(model, val_loader, loss_fn, accuracy_fn, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model salvato: {save_path}")

    print("🏁 Fine training.")
