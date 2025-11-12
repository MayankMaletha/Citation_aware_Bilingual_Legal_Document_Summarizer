# evaluation/eval_rouge.py
import json, argparse
from pathlib import Path
from evaluate import load

def main(json_dir, ref_field="gold_summary_en", hyp_field="summary_en_ctxaware"):
    rouge = load("rouge")
    refs, hyps = [], []
    for fp in Path(json_dir).glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        ref = data.get(ref_field) or data.get("summary_en")  # fallback if you put gold later
        hyp = data.get(hyp_field) or data.get("summary_en")
        if ref and hyp:
            refs.append(ref)
            hyps.append(hyp)
    if not refs:
        print("No comparable pairs found.")
        return
    score = rouge.compute(predictions=hyps, references=refs, rouge_types=["rouge1","rougeL"])
    print(score)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("json_dir")
    p.add_argument("--ref_field", default="gold_summary_en")
    p.add_argument("--hyp_field", default="summary_en_ctxaware")
    args = p.parse_args()
    main(args.json_dir, ref_field=args.ref_field, hyp_field=args.hyp_field)
