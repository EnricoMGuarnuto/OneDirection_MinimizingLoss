import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch


def get_model(cfg, mode="classification"):
    model_name = cfg["model"]["name"]
    output_dim = cfg["dataset"]["output_dim"]
    use_pretrained = cfg["pretrained"].get("pre_t", True)
    checkpoint_path = cfg["pretrained"].get("load", "")

    if model_name == "ResNet50":
        # Usa pesi ImageNet se pre_t è True
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)

        if mode == "classification":
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, output_dim)

        elif mode == "retrieval":
            modules = list(model.children())[:-1]  # tutto tranne fc
            model = nn.Sequential(*modules, nn.Flatten())

        # Carica checkpoint se specificato
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"✅ Checkpoint caricato da: {checkpoint_path}")
            except FileNotFoundError:
                print(f"⚠️ Checkpoint non trovato: {checkpoint_path}. Uso solo pesi ImageNet.")

        return model

    raise NotImplementedError(f"Model {model_name} non supportato per mode {mode}.")
