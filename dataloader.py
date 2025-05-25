import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import json


# ========================
# Custom Dataset per cartelle generiche
# ========================
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.image_paths, self.labels = [], []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for img in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img))
                self.labels.append(self.class_to_idx[cls])

        with open('class_mapping.json', 'w') as f:
            json.dump({v: k for k, v in self.class_to_idx.items()}, f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ========================
# Main Dataloader Function
# ========================
def create_dataloader(cfg, mode='train'):
    root_dir = cfg["dataset"]["path_root"]
    batch_size = cfg["data"]["batch_size_train"] if mode == 'train' else cfg["data"]["batch_size_test"]
    img_size = cfg["data"].get("img_size", 224)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    dataset_name = cfg["dataset"]["name"]
    init_args = cfg["dataset"].get(f"{mode}_init_args", cfg["dataset"].get("init_args", {}))
    
    # Se vogliamo usare un dataset torchvision
    if dataset_name in datasets.__dict__:
        DatasetClass = getattr(datasets, dataset_name)
        dataset = DatasetClass(root=root_dir, transform=transform, download=True, **init_args)
    else:
        dataset = CustomImageDataset(root_dir, transform=transform)

    if mode == 'train':
        # Se è specificato val_split, facciamo random split
        val_split = cfg["data"].get("val_split", 0)
        if val_split > 0:
            total_size = len(dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            return train_loader, val_loader
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            return train_loader, None

    elif mode == 'val':
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return val_loader

    elif mode == 'test':
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader

    else:
        raise ValueError(f"Mode '{mode}' non supportato. Usa 'train', 'val' o 'test'.")
