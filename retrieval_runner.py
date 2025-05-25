import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import get_model
from retrieval import extract_features, compute_topk, save_json
from retrieval import CustomRetrievalDataset, custom_collate, compute_metrics

def get_dataset_class_by_name(name):
    from torchvision import datasets
    try:
        return getattr(datasets, name)
    except AttributeError:
        raise NotImplementedError(f"Dataset '{name}' non trovato in torchvision.datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_queries", type=int, default=5)
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

    dataset = CustomRetrievalDataset(base_dataset)

    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size_test"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=custom_collate)

    # Attiva finetune o no
    if cfg.get("finetune", False):
        cfg["pretrained"]["load"] = cfg["training"]["save_name"]
        cfg["pretrained"]["pre_t"] = True

    model = get_model(cfg, mode="retrieval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, filenames = extract_features(model, loader, device)

    from retrieval import compute_optimal_k
    optimal_k = compute_optimal_k(dataset)
    k_adjusted = optimal_k

    print(f"🔍 Numero ottimale di top-k per classe (senza query): {optimal_k}")
    print(f"🔍 Top-k finale usato (con margine): {k_adjusted}")

    results = compute_topk(features, filenames, k=k_adjusted)
    metrics = compute_metrics(results, dataset)

    save_dir = "retrieval_results"
    model_type = "finetuned" if cfg.get("finetune", False) else "imagenet"
    filename = f"{cfg['model']['name'].lower()}_{dataset_name}_{model_type}.json"
    save_path = f"{save_dir}/{filename}"

    save_json(results, metrics, dataset, save_path)

    print(f"🔍 Config finetune: {cfg.get('finetune', False)}")
    print(f"🔍 Carico pesi da: {cfg['pretrained']['load']}")


    print(f"✅ Retrieval completato. Risultati salvati in {save_path}")
    print(f"Top-{k_adjusted} Accuracy: {metrics[0]:.4f}, Precision@{k_adjusted}: {metrics[1]:.4f}, Recall@{k_adjusted}: {metrics[2]:.4f}, mAP: {metrics[3]:.4f}, Precision@1: {metrics[4]:.4f}")

