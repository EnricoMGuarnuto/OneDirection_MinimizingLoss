import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
from collections import Counter
import yaml
from models import get_model


def custom_collate(batch):
    images = torch.stack([x[0] for x in batch])
    names = [x[1] for x in batch]
    return images, names


class CustomRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labels = []
        self.filenames = []
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            self.labels.append(label)
            self.filenames.append(str(idx))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img, self.filenames[idx]

    def get_label(self, filename):
        return self.labels[int(filename)]


def get_dataset_class_by_name(name):
    try:
        return getattr(datasets, name)
    except AttributeError:
        raise NotImplementedError(f"Dataset '{name}' non trovato in torchvision.datasets")


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


def save_json(results, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metrics": {
            "topk_accuracy": metrics[0],
            "precision@k": metrics[1],
            "recall@k": metrics[2],
            "mAP": metrics[3],
            "precision@1": metrics[4]
        }, "results": results}, f, indent=2)


def plot_retrieval(query_idx, dataset, results, num_queries=1):
    max_retrieved = min(10, len(results[0]["samples"]))
    fig, axes = plt.subplots(num_queries, max_retrieved + 1, figsize=(4 * (max_retrieved + 1), 4 * num_queries))

    if num_queries == 1:
        axes = [axes]

    for row_idx in range(num_queries):
        query_img, _ = dataset[row_idx]
        retrieved = results[row_idx]["samples"][:max_retrieved]
        axes[row_idx][0].imshow(query_img.permute(1, 2, 0))
        axes[row_idx][0].set_title("Query")
        axes[row_idx][0].axis("off")

        for i, fname in enumerate(retrieved):
            for j in range(len(dataset)):
                if dataset[j][1] == fname:
                    img, _ = dataset[j]
                    axes[row_idx][i + 1].imshow(img.permute(1, 2, 0))
                    axes[row_idx][i + 1].set_title(f"Top {i+1}")
                    axes[row_idx][i + 1].axis("off")
                    break

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size (e.g. 100 for fast tests)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--extra_margin", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="imagenet", choices=["imagenet", "finetuned"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    DatasetClass = get_dataset_class_by_name(dataset_name)
    dataset_args = dataset_cfg.get("init_args", {})
    data_path = dataset_cfg["path_root"]

    base_dataset = DatasetClass(
        root=data_path,
        transform=transform,
        download=True,
        **dataset_args
    )

    if args.limit:
        base_dataset = torch.utils.data.Subset(base_dataset, list(range(min(args.limit, len(base_dataset)))))
        
    dataset = CustomRetrievalDataset(base_dataset)

    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size_test"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=custom_collate)

    # Gestione pretrained/fine-tuned
    if args.model_type == "finetuned":
        cfg["pretrained"]["load"] = cfg["training"]["save_name"]
        cfg["pretrained"]["pre_t"] = True
    elif args.model_type == "imagenet":
        cfg["pretrained"]["load"] = ""
        cfg["pretrained"]["pre_t"] = True

    model = get_model(cfg, mode="retrieval")
    if cfg["pretrained"]["load"]:
        model.load_state_dict(torch.load(cfg["pretrained"]["load"], map_location="cpu"))
        print(f"✅ Model weights loaded from {cfg['pretrained']['load']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, filenames = extract_features(model, loader, device)

    k_adjusted = cfg["dataset"]["output_dim"] + args.extra_margin
    results = compute_topk(features, filenames, k=k_adjusted)
    metrics = compute_metrics(results, dataset)

    save_dir = "retrieval_results"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{cfg['model']['name'].lower()}_{dataset_name}_{args.model_type.lower()}.json"
    save_path = os.path.join(save_dir, filename)

    save_json(results, metrics, save_path)

    print(f"Top-{k_adjusted} Accuracy: {metrics[0]:.4f}, Precision@{k_adjusted}: {metrics[1]:.4f}, Recall@{k_adjusted}: {metrics[2]:.4f}, mAP: {metrics[3]:.4f}, Precision@1: {metrics[4]:.4f}")
    plot_retrieval(0, dataset, results, num_queries=args.n_queries)


if __name__ == "__main__":
    main()
