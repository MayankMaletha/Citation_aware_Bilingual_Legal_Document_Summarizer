# src/summarizer/citation_mini_summaries.py
import re
import logging
from typing import Dict, Any, List

from src.summarizer.summarizer import summarize_text
from src.translation.translator import translate_sentences

logger = logging.getLogger(__name__)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split(s: str) -> List[str]:
    s = (s or "").strip()
    return _SENT_SPLIT.split(s) if s else []

def _steer(entry: Dict[str, Any], body: str) -> str:
    role = entry.get("role", "MENTIONED")
    sal = entry.get("salience", 0.0)
    raw = entry.get("raw", "")
    return f"[CITATION] {raw} [ROLE={role}] [SALIENCE={sal:.2f}]\n{body}"

def summarize_citation(entry: Dict[str, Any], translate_to_hi: bool = True) -> Dict[str, Any]:
    # choose context text
    body = entry.get("context_text")
    if not body:
        if entry.get("context_window"):
            body = " ".join(entry["context_window"])
        elif entry.get("supporting_sentences"):
            body = " ".join(
                (s["sentence"] if isinstance(s, dict) else str(s))
                for s in entry["supporting_sentences"]
            )
        else:
            body = entry.get("raw", "")

    sal = float(entry.get("salience", 0.0))
    max_len = 64 if sal < 0.7 else 96 if sal < 1.0 else 128  # 👈 salience-aware length

    prompt = _steer(entry, body[:2000])

    try:
        en = summarize_text("summarize: " + prompt, max_len=max_len)
    except Exception as e:
        logger.exception("citation summarize failed: %s", e)
        en = ""

    en_sents = _split(en)

    out = {
        "citation": entry.get("citation"),
        "raw": entry.get("raw"),
        "role": entry.get("role", "MENTIONED"),
        "salience": entry.get("salience", 0.0),
        "summary_en": en,
        "summary_en_sentences": en_sents
    }

    if translate_to_hi and en_sents:
        hi_sents = translate_sentences(en_sents, src="en", tgt="hi")
        out["summary_hi_sentences"] = hi_sents
        out["summary_hi"] = " ".join(hi_sents)
    else:
        out["summary_hi_sentences"] = []
        out["summary_hi"] = ""

    return out

def summarize_all_citations(contexts: List[Dict[str, Any]], translate_to_hi: bool = True) -> List[Dict[str, Any]]:
    return [summarize_citation(c, translate_to_hi=translate_to_hi) for c in (contexts or [])]
