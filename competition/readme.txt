lista di cose da fare

1. metti il dataset nella cartella competition/data

2. imposta lo yaml:
- se solo retrieval, lascia "" in checkpoint_path e save_checkpoint
- se fai finetune, aggiungi solo la path a save_checkpoint (es. competition/saved_models/resnet50.pt)
- se fai retrieval su modello finetuned, aggiungi anche la path a checkpoint_path  (es. competition/saved_models/resnet50.pt)

3. comando per retrieval (se chiede il nome del gruppo nella funzione, altrimenti togliere ultima parte.
dipende tutto da come ci dà la funzione, al massimo chiedere a chat):
(esempio con resnet50)

python competition/retrieval.py --config competition/config/resnet50.yaml --group "One Direction-Minimizing Loss"

rmk: potrebbe non andare retrieval.py, perché la funzione che ci forniscono è diversa. chiedere a chat

4. comando per finetune

python competition/finetune.py --config competition/config/resnet50.yaml 

5. comando per retrieval del modello finetunato: uguale a retrieval sopra, basta cambiare la path nello yaml come specificato sopra.



