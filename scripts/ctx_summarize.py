# scripts/ctx_summarize.py
import json, argparse
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.summarizer.context_aware_bilingual import generate_parallel_summary
from src.summarizer.citation_mini_summaries import summarize_all_citations

def main(in_dir, out_dir=None, salience_threshold=0.45, max_contexts=8, translate_to_hi=True):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(in_dir.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))

        # ✅ Use RAW FULL TEXT, not summary_en
        raw_text = data.get("text", "")

        # ✅ SPEED FIX: Only use the most relevant textual span
        # ✅ Use already extracted *case body* text from your PDF stage
        raw_text = data.get("working_text") or data.get("summary_en") or data.get("text") or ""

        # ✅ If still empty -> fallback to concatenated context windows (always present)
        if not raw_text:
            raw_text = "\n".join(
                s["sentence"] for ctx in data.get("citation_contexts", [])
                for s in ctx.get("supporting_sentences", [])
            ) or "No content available."

# ✅ Hard truncate (VERY IMPORTANT)
        raw_text = raw_text[:3500]

        contexts = data.get("citation_contexts", [])

        # ✅ Document-level parallel summary (EN, then optional HI)
        par = generate_parallel_summary(
            working_text=raw_text,
            contexts=contexts,
            make_input_kwargs={"salience_threshold": salience_threshold, "max_contexts": max_contexts},
            max_len_en=220,               # shorter summary → more focused & faster
            translate_to_hi=translate_to_hi
        )

        # ✅ Per-citation mini explanations (appears in UI)
        cit_summaries = summarize_all_citations(contexts, translate_to_hi=translate_to_hi)

        # ✅ Save to JSON
        data["summary_en_ctxaware"] = par["summary_en"]
        data["summary_hi_ctxaware"] = par.get("summary_hi", "")
        data["summary_en_sentences"] = par.get("summary_en_sentences", [])
        data["summary_hi_sentences"] = par.get("summary_hi_sentences", [])
        data["citation_summaries"] = cit_summaries

        (out_dir / fp.name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("✔", fp.name)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("json_in_dir")
    p.add_argument("--json_out_dir", default=None)
    p.add_argument("--salience_threshold", type=float, default=0.45)
    p.add_argument("--max_contexts", type=int, default=8)
    p.add_argument("--no_translate", action="store_true")
    args = p.parse_args()

    main(args.json_in_dir, args.json_out_dir,
         salience_threshold=args.salience_threshold,
         max_contexts=args.max_contexts,
         translate_to_hi=not args.no_translate)
