import os
import argparse
import json
from collections import defaultdict

def load_mapping(mapping_file):
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    label_dict = {}
    for split in ['gallery', 'query']:
        for item in mapping[split]:
            label_dict[item['file'].replace('test/gallery/', '').replace('test/query/', '')] = item['class']
    return label_dict

def load_retrieval_results(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

def compute_topk_accuracy(results, label_dict, k):
    correct = 0
    total = len(results)
    
    for entry in results:
        query_file = entry['filename'].replace('test/query/', '')
        query_class = label_dict[query_file]
        
        retrieved_classes = [label_dict.get(f.replace('test/gallery/', ''), None) for f in entry['samples'][:k]]
        if query_class in retrieved_classes:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_json', type=str, required=True, help='Path to data_split_mapping.json')
    parser.add_argument('--results_json', type=str, required=True, help='Path to retrieval results JSON')
    parser.add_argument('--k', type=int, default=10, help='Top-k to evaluate')
    args = parser.parse_args()

    label_dict = load_mapping(args.mapping_json)
    results = load_retrieval_results(args.results_json)

    topk_acc = compute_topk_accuracy(results, label_dict, args.k)
    print(f"✅ Top-{args.k} Accuracy: {topk_acc * 100:.2f}%")

if __name__ == '__main__':
    main()
