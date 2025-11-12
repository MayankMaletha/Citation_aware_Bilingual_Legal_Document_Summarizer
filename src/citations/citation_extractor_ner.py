# src/citations/citation_extractor_ner.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = "law-ai/InLegalBERT-NER-Citation"

_tokenizer = None
_model = None
_ner = None

def get_ner():
    global _tokenizer, _model, _ner
    if _ner is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        _ner = pipeline(
            "token-classification",
            model=_model,
            tokenizer=_tokenizer,
            aggregation_strategy="simple",
            device=-1
        )
    return _ner

def extract_citations(text: str):
    """
    Returns list of dictionaries:
    { "match": <citation text>, "start": idx, "end": idx }
    """
    ner = get_ner()
    results = ner(text)
    citations = []

    for r in results:
        if r["entity_group"] in ["CASE_CITATION", "LAW_CITATION", "STATUTE"]:
            citations.append({
                "match": r["word"].strip(),
                "start": r["start"],
                "end": r["end"],
                "year": None,
                "page": None,
                "case": r["word"].strip(),
                "reporter": None,
            })

    seen = set()
    uniq = []
    for c in citations:
        key = c["match"].lower().strip()
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    return uniq
