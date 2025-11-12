# scripts/make_gold_citations.py
import json, argparse
from pathlib import Path

PREF_ORDER = ["RELIED", "OVERRULED", "MENTIONED"]

def main(indir, k=8):
    indir = Path(indir)
    for fp in sorted(indir.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        ctxs = data.get("citation_contexts", [])
        if not ctxs:
            continue
        # sort by (role preference, salience desc)
        role_rank = {r:i for i,r in enumerate(PREF_ORDER)}
        ctxs_sorted = sorted(
            ctxs,
            key=lambda c: (role_rank.get((c.get("role") or "MENTIONED").upper(), 99),
                           -float(c.get("salience", 0.0)))
        )
        gold = []
        seen = set()
        for c in ctxs_sorted:
            raw = (c.get("raw") or "").strip()
            if not raw: 
                continue
            key = " ".join(raw.lower().split())
            if key in seen:
                continue
            seen.add(key)
            gold.append(raw)
            if len(gold) >= k:
                break
        data["gold_citations"] = gold
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ gold_citations written.")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    main(args.json_dir, k=args.k)
