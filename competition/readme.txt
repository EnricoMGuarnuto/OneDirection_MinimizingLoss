lista di cose da fare

1. metti il dataset nella cartella competition/data

2. installa i requirements:

pip install -r requirements.txt

3. imposta lo yaml:  !!!
- se solo retrieval, lascia "" in checkpoint_path e save_checkpoint (tranne per mocov2: va lasciato il modello in .tar)
- se fai finetune, aggiungi solo la path a save_checkpoint (es. competition/saved_models/resnet50.pt)
- se fai retrieval su modello finetuned, aggiungi anche la path a checkpoint_path  (es. competition/saved_models/resnet50.pt)

- yaml di moco v2: 
        - per solo retrieval lasciare competition/saved_models/moco_v2_800ep_pretrain.pth.tar in checkpoint_path (togli save_checkpoint)
        - per finetune lasciare competition/saved_models/moco_v2_800ep_pretrain.pth.tar e impostare save_checkpoint
        - per fare retrieval su modello finetuned, cambia la checkpoint_path con quella del modello finetuned.

4. comando per retrieval:
(esempio con resnet50)

python competition/retrieval.py --config competition/config/resnet50.yaml 

rmk: potrebbe non andare retrieval.py, perché la funzione che ci forniscono è diversa. chiedere a chat

5. comando per finetune

python competition/finetune.py --config competition/config/resnet50.yaml 

6. comando per retrieval del modello finetunato: uguale a retrieval sopra, basta cambiare la path nello yaml come specificato sopra.



