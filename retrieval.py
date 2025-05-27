import os
import argparse
import yaml
import torch
import numpy as np
from torchvision import models
from PIL import Image
from tqdm import tqdm
import json
import timm  # use timm for flexible model loading
import torch.nn as nn
import open_clip
import torchvision.models as tv_models



def load_model(cfg, device):
    name = cfg['model']['name']
    source = cfg['model'].get('source', 'torchvision')
    pretrained = cfg['model'].get('pretrained', True)
    checkpoint_path = cfg['model'].get('checkpoint_path', '')
    num_classes = cfg['model'].get('num_classes', 1000)

    if name == 'moco_resnet50':
        model = tv_models.resnet50(pretrained=False)
        model.fc = nn.Identity()  # ⚠ rimuoviamo subito la testa (output 1000) → embedding 2048
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('module.encoder_q')}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded MoCo v2 ResNet50 from {checkpoint_path}")
        else:
            # directly a state_dict (from fine-tuned weights)
            filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('fc.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded fine-tuned weights from {checkpoint_path}")


    elif source == 'open_clip':
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained='openai')
        model = model.visual  # only visual encoder
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded custom weights from {checkpoint_path}")
        else:
            print(f"✅ Loaded {name} with pretrained weights from open_clip")

    elif source == 'timm':
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('classifier.')}
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✅ Loaded custom backbone weights from {checkpoint_path}")
        model.reset_classifier(0, '')  # remove classifier head
        print(f"✅ Loaded {name} from timm")

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
                rel_path = os.path.relpath(os.path.join(root, file), folder)
                all_files.append(rel_path)
    return sorted(all_files)

def extract_embeddings(model, image_paths, root_folder, device, img_size, norm_mean, norm_std, source, name):
    if source == 'open_clip':
        import open_clip
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
        for rel_path in tqdm(image_paths, desc=f"Extracting features from {root_folder}"):
            img_path = os.path.join(root_folder, rel_path)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model(img_tensor).squeeze().cpu().numpy()
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
    model = load_model(cfg, device)

    gallery_dir = cfg['data']['gallery_dir']
    query_dir = cfg['data']['query_dir']
    img_size = cfg['data'].get('img_size', 224)
    norm_mean = cfg['data'].get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
    norm_std = cfg['data'].get('normalization', {}).get('std', [0.229, 0.224, 0.225])
    top_k = cfg['retrieval'].get('top_k', 10)
    output_json = cfg['retrieval'].get('output_json', 'retrieval_results.json')

    source = cfg['model'].get('source', 'torchvision')
    name = cfg['model']['name']

    gallery_paths = get_image_paths(gallery_dir)
    query_paths = get_image_paths(query_dir)

    gallery_embeddings = extract_embeddings(model, gallery_paths, gallery_dir, device, img_size, norm_mean, norm_std, source, name)
    query_embeddings = extract_embeddings(model, query_paths, query_dir, device, img_size, norm_mean, norm_std, source, name)

    retrieval_results = compute_topk(query_embeddings, gallery_embeddings, top_k)
    save_json(retrieval_results, output_json)

if __name__ == '__main__':
    main()
