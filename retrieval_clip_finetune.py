import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import clip
import torch.nn as nn

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512, num_classes=90):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.clip_model.encode_image(x).float()
        return self.head(x)

def get_image_paths(folder):
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, file), folder)
                all_files.append(rel_path)
    return sorted(all_files)

def extract_embeddings(model, preprocess, image_paths, root_folder, device):
    embeddings = {}
    with torch.no_grad():
        for rel_path in tqdm(image_paths, desc=f"Extracting features from {root_folder}"):
            img_path = os.path.join(root_folder, rel_path)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.clip_model.encode_image(img_tensor).squeeze().cpu().numpy()
            embeddings[rel_path] = feature / np.linalg.norm(feature)
    return embeddings

def compute_topk(query_embeddings, gallery_embeddings, k):
    gallery_files = list(gallery_embeddings.keys())
    gallery_feats = np.stack(list(gallery_embeddings.values()))
    results = []
    for query_file, query_feat in tqdm(query_embeddings.items(), desc="Computing retrieval"):
        sims = np.dot(gallery_feats, query_feat)
        topk_idx = np.argsort(sims)[-k:][::-1]
        topk_files = [gallery_files[i] for i in topk_idx]
        results.append({"filename": query_file, "samples": topk_files})
    return results

def save_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved retrieval results to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    model = CLIPClassifier(clip_model, embed_dim=512, num_classes=90).to(device)
    model.load_state_dict(torch.load(cfg['model']['checkpoint_path'], map_location=device), strict=False)
    model.eval()

    gallery_paths = get_image_paths(cfg['data']['gallery_dir'])
    query_paths = get_image_paths(cfg['data']['query_dir'])

    gallery_embeddings = extract_embeddings(model, preprocess, gallery_paths, cfg['data']['gallery_dir'], device)
    query_embeddings = extract_embeddings(model, preprocess, query_paths, cfg['data']['query_dir'], device)

    results = compute_topk(query_embeddings, gallery_embeddings, cfg['retrieval']['top_k'])
    save_json(results, cfg['retrieval']['output_json'])

if __name__ == '__main__':
    main()


