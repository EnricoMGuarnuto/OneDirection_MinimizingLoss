import os
import shutil
import random
import argparse
from tqdm import tqdm
import json

def create_split_structure(output_root):
    os.makedirs(os.path.join(output_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'gallery'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'query'), exist_ok=True)

def split_class_images(images, train_split, gallery_split, query_split):
    random.shuffle(images)
    total = len(images)
    n_train = int(total * train_split)
    n_gallery = int(total * gallery_split)
    n_query = total - n_train - n_gallery
    
    train_imgs = images[:n_train]
    gallery_imgs = images[n_train:n_train + n_gallery]
    query_imgs = images[n_train + n_gallery:]
    
    return train_imgs, gallery_imgs, query_imgs

def copy_images(images, src_class_dir, dest_class_dir, copy_mode='copy'):
    os.makedirs(dest_class_dir, exist_ok=True)
    for img in images:
        src_path = os.path.join(src_class_dir, img)
        dest_path = os.path.join(dest_class_dir, img)
        if copy_mode == 'copy':
            shutil.copy2(src_path, dest_path)
        else:
            os.symlink(os.path.abspath(src_path), dest_path)

def prepare_dataset(data_root, output_root, train_split=0.7, gallery_split=0.15, query_split=0.15, copy_mode='copy'):
    random.seed(42)
    create_split_structure(output_root)
    
    mapping = {'train': [], 'gallery': [], 'query': []}
    
    class_names = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    for class_name in tqdm(class_names, desc="Processing classes"):
        src_class_dir = os.path.join(data_root, class_name)
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        train_imgs, gallery_imgs, query_imgs = split_class_images(images, train_split, gallery_split, query_split)
        
        # Copy or symlink images
        copy_images(train_imgs, src_class_dir, os.path.join(output_root, 'train', class_name), copy_mode)
        copy_images(gallery_imgs, src_class_dir, os.path.join(output_root, 'test', 'gallery', class_name), copy_mode)
        copy_images(query_imgs, src_class_dir, os.path.join(output_root, 'test', 'query', class_name), copy_mode)
        
        # Record mapping
        for img in train_imgs:
            mapping['train'].append({'file': f"train/{class_name}/{img}", 'class': class_name})
        for img in gallery_imgs:
            mapping['gallery'].append({'file': f"test/gallery/{class_name}/{img}", 'class': class_name})
        for img in query_imgs:
            mapping['query'].append({'file': f"test/query/{class_name}/{img}", 'class': class_name})
    
    # Save mapping to JSON
    mapping_path = os.path.join(output_root, 'data_split_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved mapping file to {mapping_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to the original dataset root')
    parser.add_argument('--output_root', type=str, required=True, help='Path to output prepared dataset')
    parser.add_argument('--train_split', type=float, default=0.7, help='Proportion for train split')
    parser.add_argument('--gallery_split', type=float, default=0.15, help='Proportion for gallery split')
    parser.add_argument('--query_split', type=float, default=0.15, help='Proportion for query split')
    parser.add_argument('--copy_mode', type=str, choices=['copy', 'symlink'], default='copy', help='Whether to copy or symlink images')

    args, unknown = parser.parse_known_args()

    prepare_dataset(args.data_root, args.output_root, args.train_split, args.gallery_split, args.query_split, args.copy_mode)

if __name__ == "__main__":
    main()
