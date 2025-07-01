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
import open_clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLIPFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embed_dim, num_classes)

        if hasattr(self.base_model, 'vision_model'):
            self.visual_encoder = self.base_model.vision_model
            print("Detected HuggingFace CLIPModel visual encoder.")
        elif hasattr(self.base_model, 'visual'):
            self.visual_encoder = self.base_model.visual
            print("Detected OpenCLIP visual encoder.")
        else:
            raise ValueError("Base model does not have a recognized visual encoder (vision_model or visual).")

        if unfreeze_layers:
            for param in self.visual_encoder.parameters():
                param.requires_grad = True
            print("✅ Base model (visual encoder) parameters UNFROZEN for fine-tuning.")
        else:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print("❄️ Base model (visual encoder) parameters FROZEN.")

    def forward(self, pixel_values):
        if hasattr(self.base_model, 'get_image_features'):
            features = self.base_model.get_image_features(pixel_values=pixel_values)
        elif hasattr(self.base_model, 'encode_image'):
            features = self.base_model.encode_image(pixel_values)
        else:
            raise ValueError("Base model does not have a recognized image encoding method.")
        
        return self.classifier(features)

def train_model(model, dataloader, epochs, lr_base, lr_classifier, clip_processor=None, open_clip_image_transform=None, save_path=None):
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
            labels = labels.to(device) 
            pixel_values = None
            if clip_processor is not None: 
                inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
                pixel_values = inputs.pixel_values
            elif open_clip_image_transform is not None:
                pixel_values = images.to(device)
            else:
                raise ValueError("No valid image processor/transform provided for training.")

            optimizer.zero_grad()
            outputs = model(pixel_values)

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
            print(f"✅ Saved best model to {save_path} (Accuracy improved to {best_accuracy:.2f}%)")
        else:
            print(f"ℹ️ Accuracy did not improve from {best_accuracy:.2f}%. Skipping save.")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)


    initial_transform = None
    clip_processor_for_batch = None 
    open_clip_transform_pipeline = None

    base_clip_model = None
    clip_embed_dim = None

    if cfg['model']['source'] == 'huggingface':
        hf_model_name = cfg['model']['name']
        clip_processor_for_batch = AutoProcessor.from_pretrained(hf_model_name)
        base_clip_model = CLIPModel.from_pretrained(hf_model_name).to(device)
        clip_embed_dim = base_clip_model.config.projection_dim
        print(f"✅ Loaded Hugging Face CLIP model: {hf_model_name} with embedding dimension {clip_embed_dim}")

        initial_transform = transforms.Compose([
            transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
            transforms.ToTensor(),
        ])

    elif cfg['model']['source'] == 'open_clip':
        model_name = cfg['model']['name']
        pretrained_weights = cfg['model'].get('pretrained')

        base_clip_model, _, open_clip_transform_pipeline = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained_weights, 
            device=device
        )
        
        if hasattr(base_clip_model, 'visual') and hasattr(base_clip_model.visual, 'output_dim'):
            clip_embed_dim = base_clip_model.visual.output_dim
        elif hasattr(base_clip_model, 'embed_dim'):
            clip_embed_dim = base_clip_model.embed_dim
        else:
            print("⚠️ Could not find specific embedding dimension for OpenCLIP. Assuming 768.")
            clip_embed_dim = 768 

        print(f"✅ Loaded OpenCLIP model: {model_name} with pretrained weights: {pretrained_weights}, embedding dimension {clip_embed_dim}")

        initial_transform = open_clip_transform_pipeline

    else:
        raise ValueError(f"Unsupported model source: {cfg['model']['source']}. Use 'huggingface' or 'open_clip'.")

    full_dataset = datasets.ImageFolder(root=cfg['data']['train_dir'], transform=initial_transform)
    num_classes = len(full_dataset.classes)
    
    dataloader = DataLoader(full_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    print(f"Dataset loaded from {cfg['data']['train_dir']} with {num_classes} classes.")

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
        clip_processor=clip_processor_for_batch,
        open_clip_image_transform=open_clip_transform_pipeline,
        save_path=cfg['training']['save_checkpoint']
    )

    print("🏁 Final training completed for classification. Model saved.")
    print(f"Model saved to: {cfg['training']['save_checkpoint']}")
    print("\nNext steps: For image retrieval, you will need to load this fine-tuned model, extract features (embeddings) before the final classifier, and then perform similarity search on your test set.")


if __name__ == '__main__':
    main()