# Image Retrieval System — Competition 2025

## Description

This repository contains the code developed for the **Competition 2025** of the *Introduction to Machine Learning* course.
The goal of the competition is to implement an **image retrieval** system that, given a set of test images, retrieves the *k* most similar images from the training set.

We adopted an approach based on **fine-tuning pretrained models** and a shared retrieval module.

---

## Project Structure

```
competition/
│
├── config/                  # YAML configuration files for each model
│   ├── clip_vitL14.yaml
│   ├── eva01_g14.yaml
│   └── resnet50.yaml
│
├── data/                    # Dataset folders
│   ├── train/            # Training images
│   └── test/             # Test images
│       ├── query/        # Query images for retrieval
│       └── gallery/      # Gallery images for retrieval
│
├── results/                 # JSON retrieval results for each model
│   ├── results_clip_vitL14_finetuned.json
│   ├── results_clip_vitL14.json
│   ├── results_eva01_g14.json
│   ├── results_resnet50_finetuned.json
│   └── results_resnet50.json
│
├── saved_models/            # Fine-tuned models (not included in public repo)
│
├── finetune_clip.py         # Fine-tuning script for CLIP ViT-L/14
├── finetune_eva01.py        # Fine-tuning script for EVA-01-G/14
├── finetune_resnet.py       # Fine-tuning script for ResNet50
│
├── retrieval.py             # Image retrieval pipeline (shared among models)
├── submit.py                # Script to generate submission JSON
│
├── requirements.txt         # Required libraries
└── README.md             # This file
```

---

## Configuration

All fine-tuning and retrieval scripts rely on `.yaml` configuration files specifying:

* Base model
* Training dataset
* Training parameters (learning rate, batch size, etc.)
* Path to save the trained models

---

## Models Used

* **ResNet50** (torchvision)
* **CLIP ViT-L/14** (HuggingFace)
* **EVA-01-G/14** (OpenCLIP)

Resnet50 and CLIP Vit-L/14 models are fine-tuned separately using their respective YAML configuration.

---

## Retrieval Pipeline

The retrieval pipeline is shared across all models:

1. Feature extraction from training and test images
2. Automatic feature saving for fast reuse
3. Distance computation between features (Cosine)
4. Selection of the *k* most similar images
5. Saving results in a JSON file, in the format required by the competition:

```json
[
  {
    "filename": "test_image.png",
    "samples": ["train_image1.png", "train_image2.png", ...]
  }
]
```

---

## Usage

### Fine-tuning:

```bash
python finetune_resnet.py --config config/resnet50.yaml
python finetune_clip.py --config config/clip_vitL14.yaml
python finetune_eva01.py --config config/eva01_g14.yaml
```

### Retrieval:

```bash
python retrieval.py --config config/resnet50.yaml
```

### Submission:

```bash
python submit.py
```

---

## Metrics

* **Top-k Accuracy** (official metric of the competition)
- Top-1 Accuracy (600 points): the correct identity is the first retrieved image.
- Top-5 Accuracy (300 points): the correct identity appears in the top-5 retrieved images.
- Top-10 Accuracy (100 points): the correct identity appears in the top-10 retrieved images

---

## Requirements

* torch
* torchvision
* tqdm
* PyYAML
* scikit-learn
* numpy
* Pillow
* open_clip_torch
* optuna


Install everything with:

```bash
pip install -r requirements.txt
```


