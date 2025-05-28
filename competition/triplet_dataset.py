import os
import random
from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images = {}
        self.all_classes = os.listdir(root_dir)

        for cls in self.all_classes:
            cls_path = os.path.join(root_dir, cls)
            images = [
                os.path.join(cls_path, img) for img in os.listdir(cls_path)
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if len(images) >= 2:
                self.class_to_images[cls] = images

        self.all_classes = list(self.class_to_images.keys())
        self.all_images = [(cls, img) for cls, imgs in self.class_to_images.items() for img in imgs]

        if not self.all_classes:
            raise ValueError("⚠ No classes with at least two images found in the dataset!")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_cls, anchor_path = self.all_images[idx]
        positive_path = random.choice([p for p in self.class_to_images[anchor_cls] if p != anchor_path])
        negative_cls = random.choice([c for c in self.all_classes if c != anchor_cls])
        negative_path = random.choice(self.class_to_images[negative_cls])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative



    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_cls, anchor_path = self.all_images[idx]
        positive_path = random.choice([p for p in self.class_to_images[anchor_cls] if p != anchor_path])
        negative_cls = random.choice([c for c in self.all_classes if c != anchor_cls])
        negative_path = random.choice(self.class_to_images[negative_cls])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
