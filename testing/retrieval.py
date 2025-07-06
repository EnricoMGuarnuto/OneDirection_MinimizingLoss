import os
import argparse
import yaml
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import json
import open_clip
from transformers import AutoProcessor, CLIPModel
from torchvision import transforms, models as tv_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom dataset to load images and apply transforms
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), os.path.basename(img_path)


# Load model and setup preprocessing & feature extraction
def load_model(cfg, device):
    source = cfg['model']['source']
    name = cfg['model']['name']
    pretrained = cfg['model'].get('pretrained', None)
    checkpoint_path = cfg['model'].get('checkpoint_path', None)

    if source == 'huggingface':
        base_model = CLIPModel.from_pretrained(name).to(device)
        processor = AutoProcessor.from_pretrained(name)
        transform = transforms.Compose([
            transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
            transforms.ToTensor(),
        ])
        feature_fn = lambda images: base_model.get_image_features(pixel_values=images)

    elif source == 'open_clip':
        base_model, _, transform = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=device
        )
        processor = None
        feature_fn = lambda images: base_model.encode_image(images)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            base_model.visual.load_state_dict(state_dict, strict=False)
            print(f"Loaded fine-tuned checkpoint from {checkpoint_path}")

    elif source == 'torchvision':
        model_fn = getattr(tv_models, name)
        base_model = model_fn(pretrained=pretrained).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)

            # Remove classifier weights to avoid dimension mismatch
            for key in ['fc.weight', 'fc.bias']:
                if key in state_dict:
                    del state_dict[key]

            base_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path} (fc layer skipped)")
        
        # Remove the classification head to get feature embeddings
        if hasattr(base_model, 'fc'):
            base_model.fc = torch.nn.Identity()
        elif hasattr(base_model, 'classifier'):
            base_model.classifier = torch.nn.Identity()

        processor = None
        transform = transforms.Compose([
            transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        feature_fn = lambda images: base_model(images)

    else:
        raise ValueError(f"Unsupported model source: {source}")

    return base_model.eval(), processor, transform, feature_fn


# Get image file paths from a folder
def get_image_paths(folder):
    exts = ['jpg', 'jpeg', 'png']
    image_paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(tuple(exts)):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)


# Extract embeddings from images using the given model
def extract_embeddings(model, dataloader, processor, feature_fn, source):
    all_embeddings = []
    all_filenames = []

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            if source == 'huggingface' and processor:
                inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
                features = feature_fn(inputs.pixel_values)
            else:
                features = feature_fn(images)

            features = torch.nn.functional.normalize(features, dim=1)
            all_embeddings.append(features.cpu())
            all_filenames.extend(paths)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_embeddings, all_filenames


# Compute top-k similar images using cosine similarity
def compute_topk(query_embeds, gallery_embeds, query_files, gallery_files, k):
    sims = cosine_similarity(query_embeds, gallery_embeds)
    results = {}

    for idx, qfile in enumerate(query_files):
        topk_idx = sims[idx].argsort()[-k:][::-1]
        topk_files = [gallery_files[i] for i in topk_idx]
        results[qfile] = topk_files
    return results

from collections import defaultdict

def evaluate_topk(results, query_paths, gallery_paths, k):
    """Valuta la top-k accuracy utilizzando i nomi delle cartelle come label."""
    # Extract labels from paths: assumes the folder name represents the class
    query_labels = {os.path.basename(p): os.path.basename(os.path.dirname(p)) for p in query_paths}
    gallery_labels = {os.path.basename(p): os.path.basename(os.path.dirname(p)) for p in gallery_paths}

    correct = 0
    total = 0

    for qfile, topk_files in results.items():
        q_label = query_labels[qfile]
        topk_labels = [gallery_labels[f] for f in topk_files]
        if q_label in topk_labels:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\n🔍 Top-{k} Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

def evaluate_retrieval_ratio(results, query_paths, gallery_paths):
    """
    Per ogni query, calcola il numero di immagini correttamente recuperate
    rispetto al totale disponibile per quella classe nella gallery.
    """
    query_labels = {os.path.basename(p): os.path.basename(os.path.dirname(p)) for p in query_paths}
    gallery_labels = {os.path.basename(p): os.path.basename(os.path.dirname(p)) for p in gallery_paths}

    # Build a map: class → set of images in the gallery
    class_to_gallery = {}
    for fname, label in gallery_labels.items():
        class_to_gallery.setdefault(label, set()).add(fname)

    total_ratio = 0
    for qfile, topk_files in results.items():
        q_label = query_labels[qfile]
        gallery_images_of_class = class_to_gallery.get(q_label, set())

    # Intersection between top-k images and those from the same class
        correct_hits = len(set(topk_files) & gallery_images_of_class)
        total_possible = len(gallery_images_of_class)

        if total_possible > 0:
            ratio = correct_hits / total_possible
            total_ratio += ratio

    average_ratio = total_ratio / len(results)
    print(f"\n📈 Retrieval accuracy (class match rate over top-k): {average_ratio:.4f}")
    return average_ratio

# Main pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--group', type=str, required=True, help='Group name for submission')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model, processor, transform, feature_fn = load_model(cfg, device)
    source = cfg['model']['source']

    batch_size = cfg['data'].get('batch_size', 32)

    query_paths = get_image_paths(cfg['data']['query_dir'])
    gallery_paths = get_image_paths(cfg['data']['gallery_dir'])

    query_dataset = ImageDataset(query_paths, transform)
    gallery_dataset = ImageDataset(gallery_paths, transform)

    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Loaded {len(query_dataset)} query images, {len(gallery_dataset)} gallery images")

    query_embeds, query_files = extract_embeddings(model, query_loader, processor, feature_fn, source)
    gallery_embeds, gallery_files = extract_embeddings(model, gallery_loader, processor, feature_fn, source)

    top_k = cfg['retrieval']['top_k']
    results = compute_topk(query_embeds, gallery_embeds, query_files, gallery_files, top_k)

    output_path = cfg['retrieval']['output_json']
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved retrieval results to {output_path}")
    evaluate_topk(results, query_paths, gallery_paths, top_k)
    evaluate_retrieval_ratio(results, query_paths, gallery_paths)

if __name__ == '__main__':
    main()
