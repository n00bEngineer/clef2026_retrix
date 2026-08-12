import json
import re
import emoji

def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Processa riga per riga (JSON lines o array grande)
with open('../data/expanded_queries_bge_large.json', 'r', encoding='utf-8') as infile, \
        open('expanded_queries_bge_large_pulito.json', 'w', encoding='utf-8') as outfile:

    data = json.load(infile)  # Se l'array intero sta in memoria
    for record in data:
        if 'original' in record:
            record['original'] = clean_tweet(record['original'])
    json.dump(data, outfile, ensure_ascii=False, indent=2)