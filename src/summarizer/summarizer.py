# src/summarizer/summarizer.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_MT5_TOKENIZER = None
_MT5_MODEL = None
_model = None
_tok = None

def _load():
    global _model, _tok

    if _model is None:
        model_path = "google/mt5-base"  # ✅ Base model from Hugging Face

        print(f"🔹 Loading base model: {model_path}")
        _tok = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        print(f"✅ Model loaded successfully on {device}")

    return _model, _tok

import re

def fix_ocr_spacing(text: str) -> str:
    """
    Fix common OCR spacing issues such as:
    - e x t r a s p a c e s  -> extraspaces
    - l i k e  t h i s      -> likethis
    """
    if not text:
        return text
    # Remove single-letter spaced text: "t h i s" -> "this"
    text = re.sub(r"(?:(?<=\w)\s(?=\w))", "", text)
    # Remove multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def summarize_text(text, max_len=260):
    model, tok = _load()                   # Load ONCE

    # Force summarization task
    text = fix_ocr_spacing(text)
    text = re.sub(r"<extra_id_\d+>", "", text)

    # ✅ Use the same prompt style used during fine-tuning
    text = "summarize: " + text

    # Tokenize directly on same device
    device = model.device
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.1,
            early_stopping=True
        )

    return tok.decode(output[0], skip_special_tokens=True).strip()

def get_mt5(model_name: str = None):
    """
    Returns (tokenizer, model). Set env MT5_MODEL_NAME to your fine-tuned path.
    Defaults to google/mt5-base.
    """
    global _MT5_TOKENIZER, _MT5_MODEL
    if _MT5_MODEL is None or _MT5_TOKENIZER is None:
        name = model_name or os.environ.get("MT5_MODEL_NAME", "google/mt5-base")
        _MT5_TOKENIZER = AutoTokenizer.from_pretrained(name)
        _MT5_MODEL = AutoModelForSeq2SeqLM.from_pretrained(name)
        if torch.cuda.is_available():
            _MT5_MODEL = _MT5_MODEL.to("cuda")
    return _MT5_TOKENIZER, _MT5_MODEL


def make_citation_aware_input(text: str, contexts,salience_threshold: float = 0.55, max_contexts: int = 12) -> str:
    """
    Improved prompt: do NOT filter citations away.
    We include the most informative citation windows instead of salience-cutoff.
    """
    text = (text or "")[:3800]  # allow slightly more context

    # Sort by salience, but DO NOT drop low-salience citations anymore.
    key = sorted(contexts or [], key=lambda c: c.get("salience", 0.0), reverse=True)[:max_contexts]

    lines = []
    for c in key:
        role = c.get("role", "MENTIONED")
        raw = c.get("raw", "")
        lines.append(f"[{role}] {raw}")

        # Add only top 2 strongest supporting sentences, not all
        supports = c.get("supporting_sentences") or []
        supports = supports[:2]
        for s in supports:
            sent = s["sentence"] if isinstance(s, dict) else str(s)
            if sent and len(sent) > 5:
                lines.append(f" - {sent.strip()}")

    return (
        "### FACTS ###\n" + text +
        "\n\n### KEY CITATION EVIDENCE ###\n" + ("\n".join(lines) if lines else "None") +
        "\n\n### TASK ###\n"
        "Write a concise legal summary (4-8 sentences). "
        "Explicitly mention key precedents when they influenced reasoning. "
        "Summarize holdings, not procedural details."
    )
