import os
import argparse
import yaml
import torch
import numpy as np
from torchvision import models
from PIL import Image
from tqdm import tqdm
import json
import timm
import torch.nn as nn
import open_clip
import torchvision.models as tv_models
from submit import submit 
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms




class ImageDataset(Dataset):
    def __init__(self, image_paths, root_folder, transform):
        self.image_paths = image_paths
        self.root_folder = root_folder
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.root_folder, rel_path)
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), rel_path

def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')
    num_classes = cfg['model'].get('num_classes', 1000)

    if source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.visual  # only visual encoder
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    else:
        model_fn = getattr(tv_models, name)
        model = model_fn(pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('classifier.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded custom backbone weights from {checkpoint_path}")
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        print(f"✅ Loaded {name} from torchvision")

    return model.to(device)

def get_image_paths(folder):
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(file)
    return sorted(all_files)

def extract_embeddings(model, image_paths, root_folder, device, img_size, norm_mean, norm_std, source, name, pretrained_tag, batch_size):
    if source == 'open_clip':
        _, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained_tag)
    else:
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    dataset = ImageDataset(image_paths, root_folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(dataloader, desc=f"Extracting features from {root_folder}"):
            batch_imgs = batch_imgs.to(device)
            feats = model(batch_imgs).cpu().numpy()
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            for path, feat in zip(batch_paths, feats):
                embeddings[path] = feat
    return embeddings

def compute_topk(query_embeddings, gallery_embeddings, k):
    gallery_files = list(gallery_embeddings.keys())
    gallery_feats = np.stack(list(gallery_embeddings.values()))
    results = {}
    for query_file, query_feat in tqdm(query_embeddings.items(), desc="Computing retrieval"):
        sims = np.dot(gallery_feats, query_feat)
        topk_idx = np.argsort(sims)[-k:][::-1]
        topk_files = [gallery_files[i] for i in topk_idx]
        results[query_file] = topk_files
    return results

def save_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved retrieval results to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--group', type=str, required=True, help='Group name for submission')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, device)

    gallery_dir = cfg['data']['gallery_dir']
    query_dir = cfg['data']['query_dir']
    img_size = cfg['data'].get('img_size', 224)
    batch_size = cfg['data'].get('batch_size', 1)
    norm_mean = cfg['data'].get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
    norm_std = cfg['data'].get('normalization', {}).get('std', [0.229, 0.224, 0.225])
    top_k = cfg['retrieval'].get('top_k', 10)
    output_json = cfg['retrieval'].get('output_json', 'retrieval_results.json')

    source = cfg['model'].get('source', 'torchvision')
    name = cfg['model']['name']
    
    pretrained = cfg['model']['pretrained']

    gallery_paths = get_image_paths(gallery_dir)
    query_paths = get_image_paths(query_dir)

    gallery_embeddings = extract_embeddings(model, gallery_paths, gallery_dir, device, img_size, norm_mean, norm_std, source, name, pretrained, batch_size)
    query_embeddings = extract_embeddings(model, query_paths, query_dir, device, img_size, norm_mean, norm_std, source, name, pretrained, batch_size)

    retrieval_results = compute_topk(query_embeddings, gallery_embeddings, top_k)
    save_json(retrieval_results, output_json)

    submit(retrieval_results, args.group)

if __name__ == '__main__':
    main()