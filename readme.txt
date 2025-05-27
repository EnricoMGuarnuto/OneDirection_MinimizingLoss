# README - Retrieval Pipeline Commands

## SETUP:

Install dependencies:

pip install timm git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git) pyyaml tqdm

If using conda:

conda install pytorch torchvision torchaudio -c pytorch
pip install timm git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git) pyyaml tqdm

## PREPARE DATASET:

python prepare\_dataset.py --data\_root /Users/enricomariaguarnuto/Desktop/peppe/OneDirection\_MinimizingLoss/data/big\_animals/animals/animals --output\_root /Users/enricomariaguarnuto/Desktop/peppe/OneDirection\_MinimizingLoss/data/new\_animals --train\_split 0.7 --gallery\_split 0.15 --query\_split 0.15 --copy\_mode copy

## RUN RETRIEVAL:

Example for ResNet50:

python retrieval.py --config config/resnet50\_v1.yaml

This will produce something like:
results\_resnet50v1.json

## COMPUTE METRICS:

python compute_metrics.py \--mapping_json data/new_animals/data_split_mapping.json \--results_json results_resnet101_finetuned.json \--k 10
## NOTES:

* Always update gallery\_dir and query\_dir inside the YAML configs to absolute paths.
* To test other models, prepare or modify the YAML files in the config/ folder.




Chiara:
-- RESNET(50,101,152) --> tutto ok 

metriche:

-Resnet50
Finetuned: 75.93
Non finetuned: 67.53

-Resnet101
Finetuned: 77.78
Non finetuned: 70.74

-Resnet152
Finetuned: 82.59
Non finetuned: 72.22

RMK: yaml unico per finetune e non finetune. quando si fa retrieval, ricordarsi di modificare:
- il nome del file json del risultato (infondo),
- le due checkpoint paths: no paths per solo retrieval, sì path per retrieval modello finetuned


-- CLIP --> non finetuned ok, accuracy = 99.26
impo: installare clip!!!!   -->   pip install git+https://github.com/openai/CLIP.git

yaml divisi:
clip.yaml --> solo retrieval
clip_finetune.yaml --> per finetuning
clip_retrieval_finetune.yaml --> per rerieval con modello finetunato

py retrieval diversi:
retrieval_clip.py --> per solo retrieval
retrieval_clip_finetune.py --> per retrieval del modello finetuned 

CLIP FINETUNED: parlando con chat, non funziona bene con cross-entropy. quindi ho usato triplet loss.
accuracy diminuisce: 98.64

-- MoCo v2 
