# scripts/build_training_data.py
import json, re, argparse
from pathlib import Path
from tqdm import tqdm

from src.summarizer.summarizer import make_citation_aware_input

def clean(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def build_input(doc_json: dict, texts_dir: Path):
    # prefer full text if available
    txt_path = texts_dir / (doc_json.get("doc_id","") + ".txt")
    if txt_path.exists():
        full_text = txt_path.read_text(encoding="utf-8")
    else:
        # fallback: try any text-like fields you saved; otherwise use summary_en_ctxaware or summary_en
        full_text = doc_json.get("full_text","") or doc_json.get("summary_en_ctxaware","") or doc_json.get("summary_en","")
    contexts = doc_json.get("citation_contexts", [])
    return make_citation_aware_input(full_text, contexts)

def main(json_dir, out_path):
    json_dir = Path(json_dir)
    texts_dir = json_dir.parent / "texts"
    items = []
    files = sorted(json_dir.glob("*.json"))
    for fp in tqdm(files):
        data = json.loads(fp.read_text(encoding="utf-8"))
        # pick best available target (prefer ctx-aware if you generated it)
        target = clean(data.get("summary_en_ctxaware") or data.get("summary_en"))
        if not target or len(target.split()) < 40:
            continue
        inp = clean(build_input(data, texts_dir))
        if not inp:
            continue
        items.append({"input": inp, "target": target})

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} training items to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir", help="e.g., output_folder/json")
    ap.add_argument("out_path", help="e.g., train.jsonl")
    args = ap.parse_args()
    main(args.json_dir, args.out_path)
