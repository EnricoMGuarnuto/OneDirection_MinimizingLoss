import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import clip


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
            feature = model.encode_image(img_tensor).squeeze().cpu().numpy()
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
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    gallery_dir = cfg['data']['gallery_dir']
    query_dir = cfg['data']['query_dir']
    top_k = cfg['retrieval'].get('top_k', 10)
    output_json = cfg['retrieval'].get('output_json', 'retrieval_results.json')

    gallery_paths = get_image_paths(gallery_dir)
    query_paths = get_image_paths(query_dir)

    gallery_embeddings = extract_embeddings(model, preprocess, gallery_paths, gallery_dir, device)
    query_embeddings = extract_embeddings(model, preprocess, query_paths, query_dir, device)

    retrieval_results = compute_topk(query_embeddings, gallery_embeddings, top_k)
    save_json(retrieval_results, output_json)


if __name__ == '__main__':
    main()

