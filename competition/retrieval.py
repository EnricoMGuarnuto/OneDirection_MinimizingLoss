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
            base_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")

        # Remove classification head
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

def get_image_paths(folder):
    exts = ['jpg', 'jpeg', 'png']
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(tuple(exts))
    ])

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

def compute_topk(query_embeds, gallery_embeds, query_files, gallery_files, k):
    sims = cosine_similarity(query_embeds, gallery_embeds)
    results = {}

    for idx, qfile in enumerate(query_files):
        topk_idx = sims[idx].argsort()[-k:][::-1]
        topk_files = [gallery_files[i] for i in topk_idx]
        results[qfile] = topk_files
    return results

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

if __name__ == '__main__':
    main()
