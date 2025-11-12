

"""
Batch runner to process a folder of PDFs and produce outputs.
Run from project root:
python -m scripts.process_folder input_folder output_folder --ocr --workers 4
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import csv, json, logging

from src.utils import md5_of_file, safe_filename
from src.extractor.text_extractor import extract
from src.cleaning.cleaner import clean_text, clean_for_json
from src.translation.translator import is_devanagari, translate_sentences
from src.citations.citation_extractor import find_citations,build_contexts
from src.citations.citation_extractor_ner import extract_citations

from src.summarizer.summarizer import make_citation_aware_input, summarize_text
from src.citations.citation_salience import classify_role, compute_salience
from src.summarizer.citation_summarizer import summarize_all_citations_in_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("process_folder")

def process_single(pdf_path, out_dir, ocr=False, ocr_page_limit=None):
    pdf_path = Path(pdf_path)
    base = safe_filename(pdf_path.stem)
    outdir = Path(out_dir)
    texts_dir = outdir / "texts"; texts_dir.mkdir(parents=True, exist_ok=True)
    json_dir = outdir / "json"; json_dir.mkdir(parents=True, exist_ok=True)

    # extract raw
    res = extract(pdf_path, ocr=ocr, ocr_page_limit=ocr_page_limit)
    raw_text = res.get("text", "")

    # clean: produce readable text (with paragraphs) and single-line text (for JSON)
    text_preserve, text_single = clean_text(raw_text)

    # save text file with preserved paragraphs
    (texts_dir / f"{base}.txt").write_text(text_preserve, encoding="utf-8")

    # detect language (simple heuristic)
    lang = "hi" if is_devanagari(text_preserve) else "en"

    # If Hindi, translate sentence-by-sentence with alignment
    alignment = None
    working_text = text_single
    if lang == "hi":
        # split into sentences based on Punkt on preserved text, then translate
        from src.utils import sentence_spans
        spans = sentence_spans(text_single)
        hindi_sents = [s for (_,_,s) in spans]
        en_sents = translate_sentences(hindi_sents, src="hi", tgt="en")
        working_text = " ".join(en_sents)
        alignment = {"hindi_count": len(hindi_sents)}

    # citations
    citations = find_citations(working_text)
    contexts = build_contexts(working_text, citations, window=5, top_k=8)

    for c in contexts:
        combined_text = " ".join(c["context_window"])
        role = classify_role(combined_text)
        sal = compute_salience(c["supporting_sentences"], role)
        c["role"] = role
        c["salience"] = sal

    citation_summaries = summarize_all_citations_in_json(
        {"citation_contexts": contexts}, sentences=2, max_out_len=96, translate_to_hi=(lang=="hi")
    )

    # build citation-aware input and summarize
    cit_input = make_citation_aware_input(working_text, contexts)
    en_summary = summarize_text(cit_input)
    # translate summary back (sentence-level)
    import re
    en_summary_sents = re.split(r'(?<=[.!?])\s+', en_summary.strip())
    hi_summary_sents = translate_sentences(en_summary_sents, src="en", tgt="hi") if lang == "hi" else None
    hi_summary = " ".join(hi_summary_sents) if hi_summary_sents else None

    # prepare json output, ensure no newlines in JSON fields
    out_json = {
        "doc_id": base,
        "filename": pdf_path.name,
        "filepath": str(pdf_path.resolve()),
        "md5": res.get("md5"),
        "ocr_used": res.get("ocr_used", False),
        "language": lang,
        "word_count": len(working_text.split()),
        "citations_count": len(citations),
        "citations": citations,
        "citation_contexts": contexts,
        "summary_en": clean_for_json(en_summary),
        "summary_hi": clean_for_json(hi_summary) if hi_summary else None,
        "alignment": alignment
    }

    (json_dir / f"{base}.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Processed %s (citations=%d)", pdf_path.name, len(citations))
    return out_json

def main(input_folder, output_folder, ocr=False, workers=2, ocr_page_limit=None):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(input_folder.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in", input_folder); return

    index_rows = []
    all_contexts = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_single, p, output_folder, ocr, ocr_page_limit): p for p in pdfs}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            p = futures[fut]
            try:
                res = fut.result()
                index_rows.append(res)
                for c in res.get("citation_contexts", []):
                    all_contexts.append({"doc_id": res["doc_id"], **c})
            except Exception as e:
                logger.exception("Error processing %s: %s", p, e)

    # write index CSV
    csv_path = output_folder / "index.csv"
    fieldnames = ["doc_id","filename","filepath","md5","ocr_used","language","word_count","citations_count"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(index_rows, key=lambda x: x["doc_id"]):
            writer.writerow({
                "doc_id": r.get("doc_id",""),
                "filename": r.get("filename",""),
                "filepath": r.get("filepath",""),
                "md5": r.get("md5",""),
                "ocr_used": r.get("ocr_used",False),
                "language": r.get("language",""),
                "word_count": r.get("word_count",0),
                "citations_count": r.get("citations_count",0)
            })

    # write combined contexts jsonl
    contexts_path = output_folder / "citation_contexts.jsonl"
    with contexts_path.open("w", encoding="utf-8") as f:
        for c in all_contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Done. Outputs in", output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ocr_page_limit", type=int, default=None)
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, ocr=args.ocr, workers=args.workers, ocr_page_limit=args.ocr_page_limit)
