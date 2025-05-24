Image Retrieval Project
=======================

Questo progetto fornisce un framework modulare per Image Retrieval basato su PyTorch. Supporta modelli preaddestrati come ResNet50 e modelli fine-tuned su dataset specifici, con salvataggio dei risultati e valutazione automatica.

Cartelle e File
---------------
- config/                → Configurazioni in YAML per ogni dataset
- data/                  → Cartella con i dataset scaricati
- models.py              → Costruzione dei modelli (es. ResNet50)
- train_model.py         → Training completo e salvataggio dei pesi
- train.py               → Funzioni per singolo step di training/validazione
- retrieval.py           → Script principale per Retrieval e valutazione
- saved_models/          → Contiene i modelli fine-tuned salvati
- retrieval_results/     → Risultati in formato JSON (metriche + immagini)

Comandi da Terminale
---------------------

Retrieval con modello preaddestrato su ImageNet:
    python retrieval.py --config config/config_pets.yaml --model_type imagenet

Retrieval con modello fine-tuned:
    python retrieval.py --config config/config_pets.yaml --model_type finetuned

Retrieval su un sottoinsieme (es. 100 immagini):
    python retrieval.py --config config/config_pets.yaml --model_type imagenet --limit 100

Training di un modello:
    python main.py --config config/config_pets.yaml

Output
------
Ogni esecuzione di retrieval salva un file in:
    retrieval_results/resnet50_nomeDataset_modelType.json

Il file JSON contiene:
- Top-k accuracy
- Precision@k, Recall@k
- Mean Average Precision (mAP)
- Precision@1
- Lista delle immagini più simili trovate per ciascuna query

Dataset Supportati
------------------
Qualsiasi dataset di torchvision può essere usato, specificando nel file YAML:
    dataset:
      name: NomeClasseTorchvision
      output_dim: numero_classi
      path_root: path/ai/dati
      init_args:
        parametri_specifici: valore

Requisiti
---------
- Python >= 3.9
- torch >= 2.0
- torchvision
- matplotlib
- tqdm
- pyyaml

Installazione rapida:
    pip install -r requirements.txt

