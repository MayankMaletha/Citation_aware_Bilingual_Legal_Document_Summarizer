# src/summarizer/context_aware_bilingual.py

import re
from typing import Dict, Any, List

from src.summarizer.summarizer import summarize_text
from src.summarizer.summarizer import make_citation_aware_input
from src.translation.translator import translate_sentences
from src.summarizer.prompt_guided import build_guided_prompt


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sents(s: str) -> List[str]:
    s = (s or "").strip()
    return _SENT_SPLIT.split(s) if s else []

# src/summarizer/context_aware_bilingual.py

from src.summarizer.summarizer import make_citation_aware_input, summarize_text, fix_ocr_spacing


def generate_parallel_summary(working_text: str,
                              contexts: List[Dict[str, Any]],
                              make_input_kwargs: Dict[str, Any] = None,
                              max_len_en: int = 180,
                              translate_to_hi: bool = True):

    make_input_kwargs = make_input_kwargs or {}

    from src.summarizer.summarizer import fix_ocr_spacing
    working_text = fix_ocr_spacing(working_text)

    salience_threshold = make_input_kwargs.get("salience_threshold", 0.55)
    max_contexts = make_input_kwargs.get("max_contexts", 8)

    prompt = make_citation_aware_input(
        working_text,
        contexts,
        salience_threshold=salience_threshold,
        max_contexts=max_contexts
    )

    summary_en = summarize_text(prompt, max_len=max_len_en)
    en_sents = _split_sents(summary_en)

    res = {"summary_en": summary_en, "summary_en_sentences": en_sents}

    if translate_to_hi and en_sents:
        hi_sents = translate_sentences(en_sents, src="en", tgt="hi")
        res["summary_hi_sentences"] = hi_sents
        res["summary_hi"] = " ".join(hi_sents)
    else:
        res["summary_hi_sentences"] = []
        res["summary_hi"] = ""

    return res
