import json

# Carica il file JSON da convertire in TSV pronto per la submit in codaBench
with open('reranked_results_nemotronFTAarsen20252026_Lora_topk1000_onbiEncoderTopk1000.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Crea il file TSV
with open('en_predictions.tsv', 'w', encoding='utf-8') as f:
    # Scrivi l'intestazione
    f.write('index\tpreds\n')
    
    # Per ogni documento (chiave = indice del post, valore = lista di predizioni)
    for index, predictions in data.items():
        # Prendi solo i primi 5 valori
        top5 = predictions[:5]
        
        # Converti la lista in stringa nel formato [pred1, pred2, ...]
        preds_string = '[' + ', '.join(top5) + ']'
        
        # Scrivi la riga
        f.write(f'{index}\t{preds_string}\n')
