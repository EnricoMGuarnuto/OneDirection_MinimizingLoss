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

python compute\_metrics.py --mapping\_json /Users/enricomariaguarnuto/Desktop/peppe/OneDirection\_MinimizingLoss/data/new\_animals/data\_split\_mapping.json --results\_json /Users/enricomariaguarnuto/Desktop/peppe/OneDirection\_MinimizingLoss/results\_resnet50v1.json --k 10

## NOTES:

* Always update gallery\_dir and query\_dir inside the YAML configs to absolute paths.
* To test other models, prepare or modify the YAML files in the config/ folder.
* If you want, we can make a script to batch-run all configs automatically.

Let me know if you want that!
