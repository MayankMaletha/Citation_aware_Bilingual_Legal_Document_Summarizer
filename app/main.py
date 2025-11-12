# app/main.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # avoid TF GPU noise

from pathlib import Path
import json
from typing import Dict, Any
import re
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils import safe_filename
from src.extractor.text_extractor import extract
from src.cleaning.cleaner import clean_text
from src.translation.translator import is_devanagari, translate_sentences
from src.citations.citation_extractor import find_citations, build_contexts

try:
    from src.citations.citation_salience import classify_role, compute_salience
    _HAS_SALIENCE = True
except Exception:
    _HAS_SALIENCE = False


# ✅ Use your fine-tuned model from Drive
MODEL_PATH = "/content/drive/MyDrive/mt5-legal-best"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device)
_model.eval()


def _generate_summary_mt5(text: str, max_length: int = 320) -> str:
    """Generate summary using your fine-tuned mT5 model with repetition control and citation cleaning."""
    # 🚫 Clean excessive citation patterns before feeding model
    text = re.sub(r"\(?\d{4}\)?\s*\(?\d+\)?\s*[A-Z]{2,}\s*\d+", "", text)  # (2005) 2 SCC 16 etc.
    text = re.sub(r"\bSCC\b|\bSLT\b|\bAIR\b|\bDLT\b|\bLJ\b|\bSCW\b|\bALL\b", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=5,
            repetition_penalty=1.5,       # 🚀 reduce repetition
            no_repeat_ngram_size=3,       # 🚀 block 3-gram loops
            length_penalty=1.0,
            early_stopping=True
        )

    summary = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    summary = re.sub(r"<extra_id_\d+>", "", summary)
    summary = re.sub(r"<REG_\d+>", "", summary)
    summary = re.sub(r"\s{2,}", " ", summary).strip()
    return summary


def _compute_roles_salience(contexts):
    """Attach role & salience if module exists, otherwise defaults."""
    if not _HAS_SALIENCE:
        for c in contexts:
            c.setdefault("role", "MENTIONED")
            c.setdefault("salience", 0.5)
        return contexts

    for c in contexts:
        combined = " ".join(c.get("context_window", []) or [])
        role = classify_role(combined)
        sal = compute_salience(c.get("supporting_sentences", []), role)
        c["role"] = role
        c["salience"] = sal
    return contexts


def process_pdf_file(pdf_path: str,
                     ocr: bool = False,
                     ocr_page_limit: int | None = None,
                     salience_threshold: float = 0.55,
                     max_contexts: int = 8,
                     translate_to_hi: bool = True) -> Dict[str, Any]:
    """
    Full pipeline for a single PDF → dict with summaries & citation contexts.
    """
    pdf = Path(pdf_path)
    doc_id = safe_filename(pdf.stem)

    # 1️⃣ Extract raw text
    ext = extract(pdf, ocr=ocr, ocr_page_limit=ocr_page_limit)
    raw_text = ext.get("text", "")

    # 2️⃣ Clean text
    text_preserve, text_single = clean_text(raw_text)

    # 3️⃣ Language detection
    lang = "hi" if is_devanagari(text_preserve) else "en"

    # 4️⃣ Translation (if Hindi → English)
    working_text = text_single
    alignment = None
    if lang == "hi":
        sents = re.split(r'(?<=[।.!?])\s+', text_single.strip()) if text_single.strip() else []
        sents = [s for s in sents if s.strip()]
        en_sents = translate_sentences(sents, src="hi", tgt="en") if sents else []
        working_text = " ".join(en_sents) if en_sents else text_single
        alignment = {"hindi_count": len(sents), "en_count": len(en_sents)}

    # 5️⃣ Citations
    citations = find_citations(working_text)
    contexts = build_contexts(working_text, citations, window=3, top_k=max_contexts)
    contexts = _compute_roles_salience(contexts)

    # 6️⃣ Generate summary (English) — now citation-cleaned internally
    summary_en = _generate_summary_mt5(working_text)

    # 7️⃣ Translate summary to Hindi (optional, with chunked translation)
    summary_hi = ""
    if translate_to_hi:
        try:
            en_chunks = textwrap.wrap(summary_en, width=350)
            hi_chunks = []
            for chunk in en_chunks:
                part = translate_sentences([chunk], src="en", tgt="hi")
                hi_chunks.append(part[0] if part else "")
            summary_hi = " ".join(hi_chunks)
        except Exception as e:
            summary_hi = f"⚠️ Translation failed: {e}"

    # 8️⃣ Build result object
    result = {
        "doc_id": doc_id,
        "filename": pdf.name,
        "language_detected": lang,
        "ocr_used": bool(ext.get("ocr_used", False)),
        "md5": ext.get("md5"),
        "word_count": len((working_text or "").split()),
        "citations_count": len(citations),
        "citation_contexts": contexts,
        "summary_en_ctxaware": summary_en,
        "summary_hi_ctxaware": summary_hi,
        "alignment": alignment,
    }
    return result


def save_json(out_obj: Dict[str, Any], out_dir: str | Path) -> Path:
    """Save output JSON file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{out_obj['doc_id']}.json"
    fp.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp
