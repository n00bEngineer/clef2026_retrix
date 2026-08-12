import pandas as pd
import json

# Carica il pickle
df = pd.read_pickle("/home/fabio/WorkInProgress/SearchEngines/Homework_SE/seupd2526-retrix/code/src/main/java/unipd/se/fineTuning_reranker/data/subtask4b_collection_data.pkl")

# Salva come JSON
df.to_json('file.json', orient='records', indent=4)


