# build_qrels.py
import json

with open("en_train.json") as f:
    topics = json.load(f)

qrels = {}
for t in topics:
    qid = str(t["index"])
    pubkey = str(t["pubkey"])
    qrels[qid] = {pubkey: 1}

with open("TRAINqrels.json", "w") as f:
    json.dump(qrels, f, indent=2)

print(f"Qrels generato: {len(qrels)} query")
