# src/summarizer/citation_summarizer.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from typing import Dict, List
from .summarizer import get_mt5
from src.translation.translator import translate_sentences
import re
import logging
import torch

logger = logging.getLogger(__name__)

def _clean_text_for_model(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()[:4000]

def _role_header(entry: Dict) -> str:
    role = entry.get("role", "MENTIONED")
    sal = entry.get("salience", 0.0)
    raw = entry.get("raw", "")
    return f"[CITATION] {raw}  [ROLE={role}]  [SALIENCE={sal:.2f}]"

def summarize_context_with_mt5(context_text: str, prefix: str = "summarize:", max_out_len: int = 80) -> str:
    tokenizer, model = get_mt5()
    inp = prefix + " " + _clean_text_for_model(context_text)
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=1024)
    if next(model.parameters()).is_cuda:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = model.generate(**inputs, max_length=max_out_len, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(out[0], skip_special_tokens=True)
    return summary.strip()

def split_into_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip()) if text and text.strip() else []

def summarize_citation_entry(context_entry: Dict, sentences: int = 2, max_out_len: int = 80,
                             translate_to_hi: bool = True) -> Dict:
    # Build context text
    ctxt = context_entry.get("context_text")
    if not ctxt:
        if context_entry.get("context_window"):
            ctxt = " ".join(context_entry["context_window"])
        elif context_entry.get("supporting_sentences"):
            ctxt = " ".join(
                s.get("sentence") if isinstance(s, dict) else str(s)
                for s in context_entry["supporting_sentences"]
            )
        else:
            ctxt = context_entry.get("raw", "")

    # Add role/salience header to steer mT5
    header = _role_header(context_entry)
    steer_text = f"{header}\n{ctxt}"

    try:
        summary_en = summarize_context_with_mt5(steer_text, prefix="summarize:", max_out_len=max_out_len)
    except Exception as e:
        logger.exception("mT5 summarization failed for citation %s: %s", context_entry.get("citation"), e)
        summary_en = ""

    sents = split_into_sentences(summary_en)
    if len(sents) < sentences and summary_en and len(summary_en.split()) < (sentences * 10):
        # second pass with a bit more budget
        try:
            summary_en = summarize_context_with_mt5(steer_text, prefix="summarize:", max_out_len=max_out_len * 2)
            sents = split_into_sentences(summary_en)
        except Exception:
            pass

    sents = sents[:sentences] if sents else ([summary_en] if summary_en else [])
    summary_en_joined = " ".join(sents).strip()

    result = {
        "citation": context_entry.get("citation"),
        "raw": context_entry.get("raw"),
        "role": context_entry.get("role", "MENTIONED"),
        "salience": context_entry.get("salience", 0.0),
        "summary_en": summary_en_joined,
        "summary_en_sentences": sents
    }

    if translate_to_hi and sents:
        try:
            hi_sents = translate_sentences(sents, src="en", tgt="hi")
            result["summary_hi_sentences"] = hi_sents
            result["summary_hi"] = " ".join(hi_sents)
        except Exception as e:
            logger.exception("Translation to Hindi failed: %s", e)
            result["summary_hi_sentences"] = []
            result["summary_hi"] = ""

    return result

def summarize_all_citations_in_json(json_obj: Dict, sentences: int = 2, max_out_len: int = 80,
                                    translate_to_hi: bool = True) -> List[Dict]:
    entries = json_obj.get("citation_contexts", [])
    return [
        summarize_citation_entry(e, sentences=sentences, max_out_len=max_out_len, translate_to_hi=translate_to_hi)
        for e in entries
    ]
