# src/utils.py
import hashlib
import logging
from nltk.tokenize.punkt import PunktSentenceTokenizer
import re

logger = logging.getLogger(__name__)
_SENT_TOKENIZER = PunktSentenceTokenizer()

def md5_of_file(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def sentence_spans(text):
    """
    returns list of (start, end, sentence_text)
    """
    spans = []
    for s,e in _SENT_TOKENIZER.span_tokenize(text):
        spans.append((s,e,text[s:e].strip()))
    return spans

def is_devanagari(text, threshold=10):
    return sum(1 for ch in text if '\u0900' <= ch <= '\u097F') >= threshold

def safe_filename(s):
    # small helper to make filename-friendly doc ids
    return re.sub(r'[^0-9A-Za-z_\-\.]', '_', s)[:200]
