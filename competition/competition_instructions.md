list of things to do

1. put the dataset in the competition/data folder

2. install requirements:

pip install -r requirements.txt

3. set the yaml: !!!
- if retrieval only, leave "" in checkpoint_path and save_checkpoint 
- if you do finetune, just add path to save_checkpoint (e.g. competition/saved_models/resnet50.pt)
- if you do retrieval on finetuned model, also add path to checkpoint_path (e.g. competition/saved_models/resnet50.pt)

4. command for retrieval:
(example with resnet50)

python competition/retrieval.py --config competition/config/resnet50.yaml 

5. command for finetune

python competition/finetune.py --config competition/config/resnet50.yaml 

6. command for retrieval of finetune model: same as retrieval above, just change the path in the yaml as specified above.



