import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets # Import datasets for ImageFolder
from tqdm import tqdm

# Import from Hugging Face transformers for CLIP model and processor
from transformers import AutoProcessor, CLIPModel

# --- Configuration for device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CLIPFineTuner Class (Provided by you, with slight adjustments) ---
# Questa classe incapsula il modello CLIP pre-addestrato e aggiunge un classificatore lineare
class CLIPFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model # Sarà un oggetto transformers.CLIPModel
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Se vogliamo fare fine-tuning, sblocchiamo i parametri del modello base
        # base_model.vision_model è l'encoder visuale di CLIP (per immagini)
        if unfreeze_layers:
            # Sblocca tutti i parametri del modello visuale
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = True
            print("✅ Base model (visual encoder) parameters UNFROZEN for fine-tuning.")
        else:
            # Congela il modello base (solo feature extraction), i.e., non aggiornare i suoi pesi
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False
            print("❄️ Base model (visual encoder) parameters FROZEN.")

    def forward(self, pixel_values):
        # Utilizza il metodo get_image_features per ottenere le embedding dal modello CLIP
        features = self.base_model.get_image_features(pixel_values=pixel_values)
        return self.classifier(features)

# --- train_model Function (Provided by you, with slight adjustments) ---
def train_model(model, dataloader, epochs, lr_base, lr_classifier, clip_processor, save_path):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Creiamo due gruppi di parametri con learning rate diversi
    # Assicurati che i nomi dei parametri corrispondano alla struttura di CLIPFineTuner
    base_params = [p for n, p in model.named_parameters() if "base_model" in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': lr_base},
        {'params': classifier_params, 'lr': lr_classifier}
    ])

    # Scheduler di learning rate, si applica a tutto l'ottimizzatore
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))

    best_accuracy = -1.0 # Per salvare il modello migliore
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            images, labels = images.to(device), labels.to(device)

            # Preprocessing con clip_processor
            # `do_rescale=False` è corretto se le immagini da transforms.ToTensor() sono già in [0,1]
            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            optimizer.zero_grad()
            outputs = model(inputs.pixel_values) # Utilizza pixel_values come input

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step() # Aggiorna lo scheduler dopo ogni batch

            running_loss += loss.item()

            # Calcolo dell'accuratezza
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Salvataggio del modello migliore (opzionale, basato su accuratezza di training)
        # Per una valutazione più robusta, dovresti usare un validation set.
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path} (Accuracy improved to {best_accuracy:.2f}%)")
        else:
            print(f"ℹ️ Accuracy did not improve from {best_accuracy:.2f}%. Skipping save.")

    return model

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Data Transformations ---
    # Nota: clip_processor farà già resizing e normalizzazione interne.
    # Queste transforms qui sono per la fase iniziale di caricamento con ImageFolder
    # e per l'input a clip_processor (che si aspetta PIL Image o Tensor in [0,1])
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(), # Converte in Tensor [0,1]
        # transforms.Normalize # Rimosso: clip_processor gestirà la normalizzazione interna
    ])
    # Importante: se clip_processor gestisce la normalizzazione, NON applicarla qui due volte.
    # La tua precedente normalizzazione era per i modelli OpenCLIP "raw".
    # HuggingFace CLIPProcessor gestisce la sua normalizzazione specifica.


    # --- Dataset and DataLoader ---
    # Utilizziamo ImageFolder per i dataset di classificazione
    full_dataset = datasets.ImageFolder(root=cfg['data']['train_dir'], transform=transform)
    num_classes = len(full_dataset.classes)
    
    dataloader = DataLoader(full_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    print(f"Dataset loaded from {cfg['data']['train_dir']} with {num_classes} classes.")

    # --- Load CLIP Model and Processor ---
    # Carica il modello base CLIP da Hugging Face
    # cfg['model']['name'] dovrebbe essere un nome modello Hugging Face valido (es. 'openai/clip-vit-large-patch14')
    # Aggiungi il controllo per il nome del modello nel tuo YAML
    if cfg['model']['source'] == 'open_clip' or cfg['model']['source'] == 'huggingface':
        # Mappa i nomi open_clip a quelli Hugging Face se necessario, o usa direttamente nomi HF
        hf_model_name = cfg['model']['name'] # Assumiamo che il nome sia già compatibile HF
        
        # Inizializza il processor CLIP
        clip_processor = AutoProcessor.from_pretrained(hf_model_name)
        # Carica il modello CLIP da Hugging Face
        base_clip_model = CLIPModel.from_pretrained(hf_model_name).to(device)
        # L'embedding dimension di default per ViT-L/14 è 768
        clip_embed_dim = base_clip_model.config.projection_dim
        print(f"✅ Loaded Hugging Face CLIP model: {hf_model_name} with embedding dimension {clip_embed_dim}")

    else:
        raise ValueError(f"Unsupported model source for classification fine-tuning: {cfg['model']['source']}. Use 'open_clip' or 'huggingface'.")

    # --- Initialize Fine-Tuner Model ---
    # Passa il parametro freeze_backbone dal YAML alla classe CLIPFineTuner
    unfreeze = not cfg['training'].get('freeze_backbone', False) # True se non congelato (default False)
    model = CLIPFineTuner(base_clip_model, clip_embed_dim, num_classes, unfreeze_layers=unfreeze)
    model = model.to(device)

    # --- Training ---
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

    print("🏁 Final training completed for classification. Model saved.")
    print(f"Model saved to: {cfg['training']['save_checkpoint']}")
    print("\nNext steps: For image retrieval, you will need to load this fine-tuned model, extract features (embeddings) before the final classifier, and then perform similarity search on your test set.")


if __name__ == '__main__':
    main()