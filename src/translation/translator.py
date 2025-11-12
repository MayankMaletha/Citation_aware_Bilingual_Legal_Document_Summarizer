# src/translation/translator.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
from src.utils import is_devanagari
from typing import List

HI_TO_EN = "Helsinki-NLP/opus-mt-hi-en"
EN_TO_HI = "Helsinki-NLP/opus-mt-en-hi"
_model_cache = {}

def get_translator(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("translation", model=model, tokenizer=tok, device=-1)
    _model_cache[model_name] = pipe
    return pipe

def chunk_sentences(sentences, max_chars=384):
    chunks=[]; curr=[]; curr_len=0
    for s in sentences:
        if curr_len + len(s) > max_chars and curr:
            chunks.append(curr); curr=[s]; curr_len=len(s)
        else:
            curr.append(s); curr_len+=len(s)
    if curr: chunks.append(curr)
    return chunks

def translate_sentences(sentences: List[str], src="hi", tgt="en") -> List[str]:
    if not sentences:
        return []
    model_name = HI_TO_EN if src.startswith("hi") and tgt.startswith("en") else EN_TO_HI if src.startswith("en") and tgt.startswith("hi") else None
    if model_name is None:
        raise ValueError("Unsupported pair")
    translator = get_translator(model_name)

    # smaller chunks to avoid 512-token overflow
    chunks = chunk_sentences(sentences, max_chars=300)
    out=[]
    for ch in chunks:
        text = "\n".join(ch)

        # key fix – max_length and truncation
        res = translator(
            text,
            max_length=256,
            truncation=True,
            clean_up_tokenization_spaces=True
        )[0]["translation_text"]

        parts = res.split("\n")
        if len(parts)==len(ch):
            out.extend([p.strip() for p in parts])
        else:
            # fallback: sentence-wise splitting
            cand = re.split(r'(?<=[।.!?])\s+', res)
            if len(cand)==len(ch):
                out.extend([c.strip() for c in cand])
            else:
                for _ in ch:
                    out.append(res.strip())
    return out[:len(sentences)]
