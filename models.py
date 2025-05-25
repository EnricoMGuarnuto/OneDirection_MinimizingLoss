import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def get_model(cfg, mode="classification"):
    model_name = cfg["model"]["name"]
    output_dim = cfg["dataset"]["output_dim"]
    use_pretrained = cfg["pretrained"].get("pre_t", True)
    checkpoint_path = cfg["pretrained"].get("load", "")

    if model_name == "ResNet50":
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        base_model = models.resnet50(weights=weights)

        if mode == "classification":
            # Usa la testa per classificazione
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, output_dim)

        elif mode == "retrieval":
            # Rimuove la testa (fc), usa solo backbone fino al flatten
            modules = list(base_model.children())[:-1]
            base_model = nn.Sequential(*modules, nn.Flatten())

        else:
            raise ValueError(f"Mode {mode} non riconosciuto.")

        # Carica checkpoint se specificato
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                base_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Checkpoint caricato da: {checkpoint_path} (strict=False)")
            except FileNotFoundError:
                print(f"⚠️ Checkpoint non trovato: {checkpoint_path}. Uso solo pesi di default.")

        return base_model

    raise NotImplementedError(f"Model {model_name} non supportato per mode {mode}.")
