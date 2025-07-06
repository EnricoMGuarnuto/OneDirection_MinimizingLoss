import os
import shutil
import random
from pathlib import Path

# Path originali
base_path = Path("Animal_Image_Dataset_original/animals/animals")  # il path dove ci sono le folder degli animali
output_base = Path("dataset")

# Percentuali di split
train_split = 0.7
query_split = 0.15  # del totale
gallery_split = 0.15  # del totale

random.seed(42)  # per riproducibilità

# Crea directory
for split in ["train", "test/query", "test/gallery"]:
    for animal_dir in base_path.iterdir():
        if animal_dir.is_dir():
            os.makedirs(output_base / split / animal_dir.name, exist_ok=True)

# Processa ciascuna classe
for animal_dir in base_path.iterdir():
    if not animal_dir.is_dir():
        continue
    images = list(animal_dir.glob("*.jpg"))
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * train_split)
    n_query = int(n_total * query_split)
    
    train_images = images[:n_train]
    query_images = images[n_train:n_train + n_query]
    gallery_images = images[n_train + n_query:]

    for img in train_images:
        shutil.copy(img, output_base / "train" / animal_dir.name / img.name)
    for img in query_images:
        shutil.copy(img, output_base / "test/query" / animal_dir.name / img.name)
    for img in gallery_images:
        shutil.copy(img, output_base / "test/gallery" / animal_dir.name / img.name)

print("Dataset split completato.")