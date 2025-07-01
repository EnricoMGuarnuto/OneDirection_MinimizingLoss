import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
import glob
from PIL import Image

# Import from Hugging Face transformers for CLIP model and processor
from transformers import AutoProcessor, CLIPModel

# --- Configuration for device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CLIPFineTuner Class (MUST be identical to the one used for fine-tuning) ---
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


class RetrievalModel(nn.Module):
    def __init__(self, fine_tuned_clip_fine_tuner):
        super().__init__()
        self.fine_tuned_clip_fine_tuner = fine_tuned_clip_fine_tuner
        
        self.fine_tuned_clip_fine_tuner.eval()
        for param in self.fine_tuned_clip_fine_tuner.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.fine_tuned_clip_fine_tuner.base_model.get_image_features(pixel_values=pixel_values)
            return torch.nn.functional.normalize(features, p=2, dim=1)


def load_model(cfg, device, num_classes):
    model_name = cfg['model']['name']
    source = cfg['model'].get('source', 'huggingface')
    
    checkpoint_path = cfg['model'].get('checkpoint_path')
    
    if source != 'huggingface':
        raise ValueError(f"Sorgente modello non supportata per il recupero: {source}. Scegliere 'huggingface'.")

    base_clip_model = CLIPModel.from_pretrained(model_name)
    clip_embed_dim = base_clip_model.config.projection_dim

    full_fine_tuned_model = CLIPFineTuner(base_clip_model, clip_embed_dim, num_classes, unfreeze_layers=False)
    
   
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        full_fine_tuned_model.load_state_dict(state_dict)
        print(f"✅ Caricato modello CLIP fine-tunato da {checkpoint_path}")
    else:
        print(f"⚠️ Percorso checkpoint '{checkpoint_path}' non trovato o vuoto. Caricamento del modello base CLIP '{model_name}'.")
       

    model = RetrievalModel(full_fine_tuned_model)

    return model.to(device)


class ImageRetrievalDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--group', type=str, required=True, help='Your group name for submission')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Using device: {device}")

    clip_processor = AutoProcessor.from_pretrained(cfg['model']['name'])
    print(f"✅ Initialized CLIP processor for {cfg['model']['name']}")

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor(),
    ])

    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    query_image_files = []
    for ext in image_extensions:
        query_image_files.extend(glob.glob(os.path.join(cfg['data']['query_dir'], ext)))
    query_image_files.sort()

    gallery_image_files = []
    for ext in image_extensions:
        gallery_image_files.extend(glob.glob(os.path.join(cfg['data']['gallery_dir'], ext)))
    gallery_image_files.sort()

    query_dataset = ImageRetrievalDataset(image_paths=query_image_files, transform=transform)
    gallery_dataset = ImageRetrievalDataset(image_paths=gallery_image_files, transform=transform)

    query_dataloader = DataLoader(query_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=4)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Loaded {len(query_dataset)} query images from {cfg['data']['query_dir']}")
    print(f"Loaded {len(gallery_dataset)} gallery images from {cfg['data']['gallery_dir']}")

    # --- Determine num_classes: prioritize YAML, otherwise infer from train_dir ---
    num_classes = cfg['data'].get('num_classes')
    if num_classes is None:
        train_dir = cfg['data'].get('train_dir')
        if train_dir and os.path.isdir(train_dir):
            num_classes = len([name for name in os.listdir(train_dir)
                               if os.path.isdir(os.path.join(train_dir, name))])
            print(f"ℹ️ 'num_classes' not found in config. Inferred from '{train_dir}': {num_classes}")
        else:
            raise ValueError(
                "Error: 'num_classes' is not specified in the config, "
                "and 'train_dir' is either missing or invalid. "
                "Please specify 'num_classes' in your config or ensure 'train_dir' points to your training dataset."
            )
    else:
        print(f"✅ 'num_classes' loaded from config: {num_classes}")

    model = load_model(cfg, device, num_classes)
    model.eval()

    print("\nExtracting Query Embeddings...")
    query_embeddings = []
    query_filepaths_basenames = []
    with torch.no_grad():
        for images, paths in tqdm(query_dataloader, desc="Query Embeddings"):
            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            embeddings = model(inputs.pixel_values)
            query_embeddings.append(embeddings.cpu())
            query_filepaths_basenames.extend([os.path.basename(p) for p in paths])
    query_embeddings = torch.cat(query_embeddings, dim=0).numpy()

    print("\nExtracting Gallery Embeddings...")
    gallery_embeddings = []
    gallery_filepaths_basenames = []
    with torch.no_grad():
        for images, paths in tqdm(gallery_dataloader, desc="Gallery Embeddings"):
            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            embeddings = model(inputs.pixel_values)
            gallery_embeddings.append(embeddings.cpu())
            gallery_filepaths_basenames.extend([os.path.basename(p) for p in paths])
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).numpy()

    print(f"Extracted {len(query_embeddings)} query embeddings and {len(gallery_embeddings)} gallery embeddings.")

    print("\nCalculating similarities...")
    similarities = cosine_similarity(query_embeddings, gallery_embeddings)
    print("Similarities calculated.")

    print("Performing retrieval...")
    res = {}
    top_k = cfg['retrieval']['top_k']

    for i, query_filename_base in enumerate(tqdm(query_filepaths_basenames, desc="Retrieving Top-K")):
        query_sims = similarities[i]
        top_k_indices = query_sims.argsort()[-top_k:][::-1]
        retrieved_gallery_filenames = [gallery_filepaths_basenames[idx] for idx in top_k_indices]
        res[query_filename_base] = retrieved_gallery_filenames
    
    print("Retrieval complete. Results compiled.")

    output_json_path = cfg['retrieval']['output_json']
    os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Results saved to {output_json_path}")

    print("🏁 Retrieval process completed.")

if __name__ == '__main__':
    main()