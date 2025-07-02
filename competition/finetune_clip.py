import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLIPFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embed_dim, num_classes)

        if unfreeze_layers:
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        features = self.base_model.get_image_features(pixel_values=pixel_values)
        return self.classifier(features)

def train_model(model, dataloader, epochs, lr_base, lr_classifier, clip_processor, save_path):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    base_params = [p for n, p in model.named_parameters() if "base_model" in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': lr_base},
        {'params': classifier_params, 'lr': lr_classifier}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))

    best_accuracy = -1.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            images, labels = images.to(device), labels.to(device)

            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            optimizer.zero_grad()
            outputs = model(inputs.pixel_values) 

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
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (Accuracy improved to {best_accuracy:.2f}%)")
        else:
            print(f"ℹAccuracy did not improve from {best_accuracy:.2f}%. Skipping save.")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(), 
    ])

    full_dataset = datasets.ImageFolder(root=cfg['data']['train_dir'], transform=transform)
    num_classes = len(full_dataset.classes)
    
    dataloader = DataLoader(full_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    print(f"Dataset loaded from {cfg['data']['train_dir']} with {num_classes} classes.")

    if cfg['model']['source'] == 'open_clip' or cfg['model']['source'] == 'huggingface':
        hf_model_name = cfg['model']['name']
        

        clip_processor = AutoProcessor.from_pretrained(hf_model_name)

        base_clip_model = CLIPModel.from_pretrained(hf_model_name).to(device)

        clip_embed_dim = base_clip_model.config.projection_dim
        print(f"Loaded Hugging Face CLIP model: {hf_model_name} with embedding dimension {clip_embed_dim}")

    else:
        raise ValueError(f"Unsupported model source for classification fine-tuning: {cfg['model']['source']}. Use 'open_clip' or 'huggingface'.")

    unfreeze = not cfg['training'].get('freeze_backbone', False)
    model = CLIPFineTuner(base_clip_model, clip_embed_dim, num_classes, unfreeze_layers=unfreeze)
    model = model.to(device)

    print("\nStarting training...")
    train_model(
        model=model,
        dataloader=dataloader,
        epochs=cfg['training']['num_epochs'],
        lr_base=float(cfg['training']['lr_backbone']),
        lr_classifier=float(cfg['training']['lr_head']),
        clip_processor=clip_processor,
        save_path=cfg['training']['save_checkpoint']
    )

    print("Final training completed for classification. Model saved.")
    print(f"Model saved to: {cfg['training']['save_checkpoint']}")
    print("\nNext steps: For image retrieval, you will need to load this fine-tuned model, extract features (embeddings) before the final classifier, and then perform similarity search on your test set.")


if __name__ == '__main__':
    main()