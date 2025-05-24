import argparse
import yaml

from models import get_model
from dataloader import get_dataloaders
from train_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path al file di configurazione .yaml")
    args = parser.parse_args()

    # Carica la configurazione
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Inizializza modello, dataloader e lancia il training
    model = get_model(cfg)
    train_loader, val_loader = get_dataloaders(cfg)
    train_model(model, train_loader, val_loader, cfg)
