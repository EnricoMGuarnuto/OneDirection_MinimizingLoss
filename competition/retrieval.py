# import os
# import argparse
# import yaml
# import torch
# import numpy as np
# from torchvision import models
# from PIL import Image
# from tqdm import tqdm
# import json
# import timm
# import torch.nn as nn
# import open_clip
# import torchvision.models as tv_models
# from submit import submit 
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms




# class ImageDataset(Dataset):
#     def __init__(self, image_paths, root_folder, transform):
#         self.image_paths = image_paths
#         self.root_folder = root_folder
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         rel_path = self.image_paths[idx]
#         img_path = os.path.join(self.root_folder, rel_path)
#         img = Image.open(img_path).convert('RGB')
#         return self.transform(img), rel_path

# def load_model(cfg, device):
#     name = cfg['model']['name']
#     source = cfg['model'].get('source', 'torchvision')
#     pretrained = cfg['model'].get('pretrained', True)
#     checkpoint_path = cfg['model'].get('checkpoint_path', '')
#     num_classes = cfg['model'].get('num_classes', 1000)

#     if source == 'open_clip':
#         model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
#         model = model.visual  # only visual encoder
#         if checkpoint_path:
#             state_dict = torch.load(checkpoint_path, map_location=device)
#             model.load_state_dict(state_dict, strict=False)
#             print(f"✅ Loaded custom weights from {checkpoint_path}")
#         else:
#             print(f"✅ Loaded {name} with pretrained weights from open_clip")

#     else:
#         model_fn = getattr(tv_models, name)
#         model = model_fn(pretrained=pretrained)
#         if checkpoint_path:
#             state_dict = torch.load(checkpoint_path, map_location=device)
#             filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.') and not k.startswith('classifier.')}
#             model.load_state_dict(filtered_state_dict, strict=False)
#             print(f"✅ Loaded custom backbone weights from {checkpoint_path}")
#         if hasattr(model, 'fc'):
#             model.fc = nn.Identity()
#         elif hasattr(model, 'classifier'):
#             model.classifier = nn.Identity()
#         print(f"✅ Loaded {name} from torchvision")

#     return model.to(device)

# def get_image_paths(folder):
#     all_files = []
#     for root, _, files in os.walk(folder):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 all_files.append(file)
#     return sorted(all_files)

# def extract_embeddings(model, image_paths, root_folder, device, img_size, norm_mean, norm_std, source, name, pretrained_tag, batch_size):
#     if source == 'open_clip':
#         _, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained_tag)
#     else:
#         preprocess = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=norm_mean, std=norm_std)
#         ])

#     dataset = ImageDataset(image_paths, root_folder, preprocess)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#     embeddings = {}
#     model.eval()
#     with torch.no_grad():
#         for batch_imgs, batch_paths in tqdm(dataloader, desc=f"Extracting features from {root_folder}"):
#             batch_imgs = batch_imgs.to(device)
#             feats = model(batch_imgs).cpu().numpy()
#             feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
#             for path, feat in zip(batch_paths, feats):
#                 embeddings[path] = feat
#     return embeddings

# def compute_topk(query_embeddings, gallery_embeddings, k):
#     gallery_files = list(gallery_embeddings.keys())
#     gallery_feats = np.stack(list(gallery_embeddings.values()))
#     results = {}
#     for query_file, query_feat in tqdm(query_embeddings.items(), desc="Computing retrieval"):
#         sims = np.dot(gallery_feats, query_feat)
#         topk_idx = np.argsort(sims)[-k:][::-1]
#         topk_files = [gallery_files[i] for i in topk_idx]
#         results[query_file] = topk_files
#     return results

# def save_json(results, output_path):
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"✅ Saved retrieval results to {output_path}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
#     parser.add_argument('--group', type=str, required=True, help='Group name for submission')
#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         cfg = yaml.safe_load(f)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = load_model(cfg, device)

#     gallery_dir = cfg['data']['gallery_dir']
#     query_dir = cfg['data']['query_dir']
#     img_size = cfg['data'].get('img_size', 224)
#     batch_size = cfg['data'].get('batch_size', 1)
#     norm_mean = cfg['data'].get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
#     norm_std = cfg['data'].get('normalization', {}).get('std', [0.229, 0.224, 0.225])
#     top_k = cfg['retrieval'].get('top_k', 10)
#     output_json = cfg['retrieval'].get('output_json', 'retrieval_results.json')

#     source = cfg['model'].get('source', 'torchvision')
#     name = cfg['model']['name']
    
#     pretrained = cfg['model']['pretrained']

#     gallery_paths = get_image_paths(gallery_dir)
#     query_paths = get_image_paths(query_dir)

#     gallery_embeddings = extract_embeddings(model, gallery_paths, gallery_dir, device, img_size, norm_mean, norm_std, source, name, pretrained, batch_size)
#     query_embeddings = extract_embeddings(model, query_paths, query_dir, device, img_size, norm_mean, norm_std, source, name, pretrained, batch_size)

#     retrieval_results = compute_topk(query_embeddings, gallery_embeddings, top_k)
#     save_json(retrieval_results, output_json)

#     submit(retrieval_results, args.group)

# if __name__ == '__main__':
#     main()


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
            print(f"✅ Loaded fine-tuned checkpoint from {checkpoint_path}")

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

    print(f"✅ Loaded {len(query_dataset)} query images, {len(gallery_dataset)} gallery images")

    query_embeds, query_files = extract_embeddings(model, query_loader, processor, feature_fn, source)
    gallery_embeds, gallery_files = extract_embeddings(model, gallery_loader, processor, feature_fn, source)

    top_k = cfg['retrieval']['top_k']
    results = compute_topk(query_embeds, gallery_embeds, query_files, gallery_files, top_k)

    output_path = cfg['retrieval']['output_json']
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved retrieval results to {output_path}")

if __name__ == '__main__':
    main()
