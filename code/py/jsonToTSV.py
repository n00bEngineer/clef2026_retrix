"""Convert the collection JSON file into a tab-separated corpus file.

The script reads `collection_data.json` from the current working directory and
writes `corpus.tsv` with the fields expected by the downstream indexing stage.
Tabs and newlines inside text fields are normalized to spaces so each document
remains a single TSV row.
"""

import json

with open("collection_data.json") as f:
    data = json.load(f)

with open("corpus.tsv", "w", encoding="utf-8") as out:
    # Keep the column order aligned with the Java-side corpus reader.
    out.write("pubkey\ttitle\tabstract\tvenue\tauthors\n")

    for item in data:
        row = [
            str(item.get("pubkey", "")),
            item.get("title", "").replace("\t", " ").replace("\n", " "),
            item.get("abstract", "").replace("\t", " ").replace("\n", " "),
            item.get("venue", "").replace("\t", " ").replace("\n", " "),
            item.get("authors", "").replace("\t", " ").replace("\n", " "),
        ]
        out.write("\t".join(row) + "\n")
