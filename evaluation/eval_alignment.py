# evaluation/eval_alignment.py
import json, argparse, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from src.translation.translator import translate_sentences

def main(json_dir, hyp_field_en="summary_en_ctxaware", hyp_field_hi="summary_hi_ctxaware"):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    sims = []
    for fp in Path(json_dir).glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        en = data.get(hyp_field_en) or data.get("summary_en")
        hi = data.get(hyp_field_hi) or data.get("summary_hi")
        if not en or not hi:
            continue
        # translate HI->EN to compare semantics
        en_hi = translate_sentences([hi], src="hi", tgt="en")[0]
        emb = model.encode([en, en_hi], convert_to_tensor=True)
        sim = float(util.cos_sim(emb[0], emb[1]).item())
        sims.append(sim)
    if not sims:
        print("No bilingual pairs found.")
        return
    print({"count": len(sims), "mean_alignment": float(np.mean(sims)), "std": float(np.std(sims))})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir")
    ap.add_argument("--en_field", default="summary_en_ctxaware")
    ap.add_argument("--hi_field", default="summary_hi_ctxaware")
    args = ap.parse_args()
    main(args.json_dir, hyp_field_en=args.en_field, hyp_field_hi=args.hi_field)
