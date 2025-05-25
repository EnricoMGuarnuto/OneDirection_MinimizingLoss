import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

class CustomRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = []
        self.filenames = []  # gli indici numerici (usati internamente)
        self.real_filenames = []  # i veri nomi file (per il JSON)

        for idx in range(len(self.base_dataset)):
            img_data = self.base_dataset[idx]
            if isinstance(img_data, tuple) and len(img_data) == 2:
                _, label = img_data
            else:
                label = None  # fallback

            self.labels.append(label)
            self.filenames.append(str(idx))

            # Cerca di estrarre il vero nome file se disponibile
            if hasattr(base_dataset, 'images'):  # torchvision ImageFolder, DTD, ecc.
                full_path = base_dataset.images[idx][0]
                filename = full_path.split("/")[-1]
            else:
                filename = f"img_{idx}.jpg"  # fallback generico

            self.real_filenames.append(filename)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img, self.filenames[idx]

    def get_label(self, filename):
        return self.labels[int(filename)]

    def get_real_filename(self, filename):
        return self.real_filenames[int(filename)]

def custom_collate(batch):
    images = torch.stack([x[0] for x in batch])
    names = [x[1] for x in batch]
    return images, names

def compute_optimal_k(dataset):
    class_counts = Counter(dataset.labels)
    max_per_class = max(class_counts.values()) - 1  # togli la query
    return max_per_class

def extract_features(model, data_loader, device):
    model.eval()
    model.to(device)
    features, filenames = [], []

    with torch.no_grad():
        for imgs, names in tqdm(data_loader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu())
            filenames.extend(names)

    return torch.cat(features), filenames

def compute_topk(features, filenames, k=5):
    n = features.size(0)
    k = min(k, n - 1)
    similarity = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    results = []
    for i in range(n):
        topk = similarity[i].topk(k + 1).indices[1:]
        results.append({
            "filename": filenames[i],
            "samples": [filenames[j] for j in topk]
        })
    return results

def save_json(results, metrics, dataset, path):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prova a capire da dove prendere i path
    if hasattr(dataset.base_dataset, 'samples'):
        get_path = lambda idx: dataset.base_dataset.samples[int(idx)][0]
    elif hasattr(dataset.base_dataset, 'imgs'):
        get_path = lambda idx: dataset.base_dataset.imgs[int(idx)][0]
    elif hasattr(dataset.base_dataset, 'image_paths'):
        get_path = lambda idx: dataset.base_dataset.image_paths[int(idx)]
    else:
        raise AttributeError("Il dataset non ha un attributo noto per recuperare i path delle immagini.")

    json_data = {
        "metrics": {
            "topk_accuracy": metrics[0],
            "precision@k": metrics[1],
            "recall@k": metrics[2],
            "mAP": metrics[3],
            "precision@1": metrics[4]
        },
        "results": [
            {
                "query": get_path(entry["filename"]),
                "retrieved": [get_path(fid) for fid in entry["samples"]]
            }
            for entry in results
        ]
    }

    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)



from collections import Counter

def compute_metrics(results, dataset):
    precision_list, recall_list, ap_list, precision_at_1_list = [], [], [], []
    correct_total = 0
    class_counts = Counter(dataset.labels)

    for entry in results:
        query_label = dataset.get_label(entry["filename"])
        retrieved_labels = [dataset.get_label(f) for f in entry["samples"]]
        relevant = [1 if label == query_label else 0 for label in retrieved_labels]

        correct = sum(relevant)
        precision = correct / len(relevant)
        recall = correct / class_counts[query_label]
        precisions = []
        correct_so_far = 0
        for i, rel in enumerate(relevant):
            if rel:
                correct_so_far += 1
                precisions.append(correct_so_far / (i + 1))
        ap = sum(precisions) / correct if correct > 0 else 0
        precision_at_1 = relevant[0] if relevant else 0

        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(ap)
        precision_at_1_list.append(precision_at_1)

        if query_label in retrieved_labels:
            correct_total += 1

    return (
        correct_total / len(results),
        sum(precision_list) / len(precision_list),
        sum(recall_list) / len(recall_list),
        sum(ap_list) / len(ap_list),
        sum(precision_at_1_list) / len(precision_at_1_list)
    )
