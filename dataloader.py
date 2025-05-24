import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import Caltech101, OxfordIIITPet, DTD
from PIL import Image
import json

# ========================
# Custom Dataset (Train)
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
# Custom Dataset (Test)
# ========================

class CustomTestImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [(os.path.join(root_dir, fname), fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path, fname = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname

# ========================
# Main Dataloader Function
# ========================

def create_dataloader(root_dir, batch_size=32, img_size=224, val_split=0.2, mode='train', transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    # Auto-select dataset
    if "caltech" in root_dir.lower():
        dataset = Caltech101(root=root_dir, download=True, transform=transform)
    elif "pets" in root_dir.lower():
        dataset = OxfordIIITPet(root=root_dir, download=True, transform=transform)
    elif "dtd" in root_dir.lower():
        dataset = DTD(root=root_dir, split='train', download=True, transform=transform)
    elif "competition/test" in root_dir.lower():
        test_dataset = CustomTestImageDataset(root_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader, None, None, None
    elif "competition/train" in root_dir.lower():
        dataset = CustomImageDataset(root_dir, transform=transform)
    else:
        raise ValueError(f"Dataset path {root_dir} non riconosciuto.")

    if mode == 'train':
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, val_loader, None, None

    elif mode == 'test':
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader, None, None, None

