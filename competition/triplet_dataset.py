# import os
# import random
# from torch.utils.data import Dataset
# from PIL import Image

# class TripletDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform

#         # Costruisci mappa solo con classi valide (almeno 2 immagini)
#         self.class_to_imgs = {}
#         raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#         for cls in raw_classes:
#             cls_path = os.path.join(root_dir, cls)
#             images = [os.path.join(cls_path, img)
#                       for img in os.listdir(cls_path)
#                       if img.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(cls_path, img))]
#             if len(images) >= 2:
#                 self.class_to_imgs[cls] = images
#             else:
#                 print(f"⚠ Classe '{cls}' scartata (solo {len(images)} immagine)")

#         self.classes = list(self.class_to_imgs.keys())
#         if len(self.classes) < 2:
#             raise ValueError("⚠ Servono almeno due classi con ≥2 immagini ciascuna per creare triplet!")

#         self.all_images = [(cls, img) for cls, imgs in self.class_to_imgs.items() for img in imgs]

#     def __len__(self):
#         return len(self.all_images)

#     def __getitem__(self, idx):
#         anchor_cls, anchor_path = self.all_images[idx]

#         positive_candidates = [p for p in self.class_to_imgs[anchor_cls] if p != anchor_path]
#         if not positive_candidates:
#             raise ValueError(f"⚠ Nessuna immagine positiva per classe {anchor_cls}")
#         positive_path = random.choice(positive_candidates)

#         negative_classes = [c for c in self.classes if c != anchor_cls]
#         if not negative_classes:
#             raise ValueError(f"⚠ Nessuna classe negativa disponibile (solo '{anchor_cls}')")
#         negative_cls = random.choice(negative_classes)
#         negative_path = random.choice(self.class_to_imgs[negative_cls])

#         anchor = Image.open(anchor_path).convert('RGB')
#         positive = Image.open(positive_path).convert('RGB')
#         negative = Image.open(negative_path).convert('RGB')

#         if self.transform:
#             anchor = self.transform(anchor)
#             positive = self.transform(positive)
#             negative = self.transform(negative)

#         return anchor, positive, negative


import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_images = {}
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:  # at least 2 needed per class for anchor-positive
                self.class_to_images[class_name] = images

        self.classes = list(self.class_to_images.keys())
        self.data = [(cls, img) for cls, imgs in self.class_to_images.items() for img in imgs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_class, anchor_path = self.data[index]
        positive_path = random.choice([p for p in self.class_to_images[anchor_class] if p != anchor_path])

        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

