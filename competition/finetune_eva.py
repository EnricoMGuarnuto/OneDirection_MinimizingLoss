import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import open_clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fine-tuning wrapper for EVA CLIP model
class EVAFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes):
        super().__init__()
        self.base_model = base_model
        self.visual_encoder = base_model.visual
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Enable gradient updates on the visual encoder
        for param in self.visual_encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.base_model.encode_image(x)
        return self.classifier(features)


# Training loop
def train_model(model, dataloader, epochs, lr_base, lr_classifier, save_path):
    criterion = nn.CrossEntropyLoss()

    backbone_params = []
    classifier_params = []

    # Separate parameters for different learning rates
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual_encoder" in name:
                backbone_params.append(param)
            elif "classifier" in name:
                classifier_params.append(param)
            elif "logit_scale" in name:
                # Skip specific parameter not meant to be trained here
                print(f"Skipping logit_scale param: {name}")
                param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_base},
        {'params': classifier_params, 'lr': lr_classifier}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    best_accuracy = -1.0

    # Main training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")

        # Save best model based on accuracy
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at {save_path} (Accuracy: {best_accuracy:.2f}%)")


# Main function for loading config, dataset, and running training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load EVA CLIP model with pretrained weights
    model_name = cfg['model']['name']
    pretrained_weights = cfg['model']['pretrained']
    base_model, _, open_clip_transform = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_weights, device=device
    )

    # Detect embedding dimension automatically
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        embedding = base_model.encode_image(dummy)
    embed_dim = embedding.shape[-1]
    print(f"Detected embedding dimension: {embed_dim}")

    # Load dataset
    dataset = datasets.ImageFolder(root=cfg['data']['train_dir'], transform=open_clip_transform)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)

    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Initialize fine-tuning model
    model = EVAFineTuner(base_model, embed_dim, num_classes).to(device)

    # Train the model
    train_model(
        model=model,
        dataloader=dataloader,
        epochs=cfg['training']['num_epochs'],
        lr_base=float(cfg['training']['lr_backbone']),
        lr_classifier=float(cfg['training']['lr_head']),
        save_path=cfg['training']['save_checkpoint']
    )


if __name__ == '__main__':
    main()
