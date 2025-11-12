# scripts/summarize_contexts.py

import json
import pandas as pd
from tqdm import tqdm

from src.summarizer.summarizer import ContextAwareSummarizer
from src.translation.translator import translate_sentences
import re


# EN → HI stable translator
from src.translation.translator import translate_sentences
import re

def en_to_hi(summary_en: str):
    sents = re.split(r'(?<=[.!?])\s+', summary_en)
    hi_sents = translate_sentences(sents, src="en", tgt="hi")
    return " ".join(hi_sents)


def load_contexts(jsonl_path):
    """
    Loads citation contexts produced by process_folder.py into a grouped format:
    {doc_id: [context objects...]}
    """
    docs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            doc_id = entry["doc_id"]
            docs.setdefault(doc_id, []).append(entry)
    return docs


def main(context_jsonl, output_csv, model="google/mt5-small"):
    summarizer = ContextAwareSummarizer(model)

    grouped_contexts = load_contexts(context_jsonl)

    rows = []

    for doc_id, contexts in tqdm(grouped_contexts.items(), desc="Summarizing"):
        # context-aware English summary
        summary_en = summarizer.summarize(contexts)

        # convert to Hindi using your translator
        summary_hi = en_to_hi(summary_en)

        rows.append({
            "doc_id": doc_id,
            "summary_en": summary_en,
            "summary_hi": summary_hi
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved: {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("context_jsonl", help="Path to citation_contexts.jsonl")
    parser.add_argument("output_csv", help="Where to save summaries")
    parser.add_argument("--model", default="google/mt5-small", help="Path or name of mT5 model")
    args = parser.parse_args()
    main(args.context_jsonl, args.output_csv, model=args.model)
