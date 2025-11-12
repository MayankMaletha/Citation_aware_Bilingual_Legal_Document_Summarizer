# evaluation/eval_citation_metrics.py  (replace existing)
import json, re, argparse
from pathlib import Path
from difflib import SequenceMatcher
try:
    from rapidfuzz import fuzz
    rapidfuzz_available = True
except Exception:
    rapidfuzz_available = False

def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\/\s]", " ", s)          # keep slashes for cases like 606/2024
    s = re.sub(r"\b(scc|air|crl\.?a|manu)\b", lambda m: m.group(1).replace(".", ""), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def similarity(a: str, b: str) -> float:
    if rapidfuzz_available:
        return fuzz.partial_ratio(a, b) / 100.0
    else:
        return SequenceMatcher(None, a, b).ratio()

def main(json_dir, hyp_field="summary_en_ctxaware", top_k=12, sim_thresh=0.55):
    json_dir = Path(json_dir)
    totals = {"tp":0, "fp":0, "fn":0, "docs":0}

    for fp in json_dir.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        contexts = sorted(data.get("citation_contexts", []), key=lambda c: c.get("salience",0), reverse=True)[:top_k]
        gold_names = [norm(c.get("raw","")) for c in contexts if c.get("raw")]
        hyp = data.get(hyp_field) or ""
        hyp_n = norm(hyp)

        matched = set()
        tp = 0
        for i,g in enumerate(gold_names):
            if not g: continue
            # exact containment passes quickly
            if g in hyp_n:
                tp += 1
                matched.add(i)
                continue
            # otherwise fuzzy / ratio
            sim = similarity(g, hyp_n)
            if sim >= sim_thresh:
                tp += 1
                matched.add(i)

        fn = len(gold_names) - tp
        # we keep fp=0 (we're checking gold set coverage). Optionally compute hallucinated citations separately.
        fp = 0

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["docs"] += 1

    recall = totals["tp"] / (totals["tp"] + totals["fn"] + 1e-9)
    precision = totals["tp"] / (totals["tp"] + totals["fp"] + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print({"docs": totals["docs"], "precision": precision, "recall": recall, "f1": f1})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir")
    ap.add_argument("--hyp_field", default="summary_en_ctxaware")
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--sim_thresh", type=float, default=0.55)
    args = ap.parse_args()
    main(args.json_dir, hyp_field=args.hyp_field, top_k=args.top_k, sim_thresh=args.sim_thresh)
