import os
import argparse
import yaml
import torch
import numpy as np
from torchvision import models
from PIL import Image
from tqdm import tqdm
import timm
import torch.nn as nn
import open_clip
import json

# Function provided by organizers (placeholder)
# def submit(results_dict, group_name):
#     # Replace with actual submission logic
#     print(f"📝 Submitting results for group: {group_name}")
#     print(f"🔢 {len(results_dict)} queries processed.")
#     # You can replace this with the real submit() function provided on competition day
#     with open(f"{group_name}_submission.json", "w") as f:
#         json.dump(results_dict, f, indent=2)
#     print(f"✅ Submission dictionary saved as {group_name}_submission.json")


def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')
    if source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    elif source == 'timm':
        model = timm.create_model(name, pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        model.reset_classifier(0)
    else:
        model_fn = getattr(models, name)
        model = model_fn(pretrained=pretrained)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        if hasattr(model, 'fc'):
            model.fc = nn.Identity()
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
    return model.to(device).eval()


def get_image_paths(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


def extract_embeddings(model, image_paths, root_folder, device, img_size, norm_mean, norm_std, source, name):
    if source == 'open_clip':
        _, _, preprocess = open_clip.create_model_and_transforms(name, pretrained='openai')
    else:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    embeddings = {}
    with torch.no_grad():
        for fname in tqdm(image_paths, desc=f"Extracting features from {root_folder}"):
            img_path = os.path.join(root_folder, fname)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model(img_tensor).squeeze().cpu().numpy()
            embeddings[fname] = feature / np.linalg.norm(feature)
    return embeddings


def compute_topk(query_embeddings, gallery_embeddings, k):
    gallery_files = list(gallery_embeddings.keys())
    gallery_feats = np.stack([gallery_embeddings[f] for f in gallery_files])
    results = {}
    for query_file, query_feat in tqdm(query_embeddings.items(), desc="Computing retrieval"):
        sims = np.dot(gallery_feats, query_feat)
        topk_idx = np.argsort(sims)[-k:][::-1]
        topk_files = [gallery_files[i] for i in topk_idx]
        results[query_file] = topk_files
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--group', type=str, required=True, help='Group name for submission')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, device)

    query_dir = cfg['data']['query_dir']
    gallery_dir = cfg['data']['gallery_dir']
    img_size = cfg['data']['img_size']
    norm_mean = cfg['data']['normalization']['mean']
    norm_std = cfg['data']['normalization']['std']
    top_k = cfg['retrieval']['top_k']

    source = cfg['model']['source']
    name = cfg['model']['name']

    query_paths = get_image_paths(query_dir)
    gallery_paths = get_image_paths(gallery_dir)

    query_embeddings = extract_embeddings(model, query_paths, query_dir, device, img_size, norm_mean, norm_std, source, name)
    gallery_embeddings = extract_embeddings(model, gallery_paths, gallery_dir, device, img_size, norm_mean, norm_std, source, name)

    results = compute_topk(query_embeddings, gallery_embeddings, top_k)
    submit(results, args.group)


if __name__ == '__main__':
    main()
