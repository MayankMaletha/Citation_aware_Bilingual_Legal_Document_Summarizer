# src/citations/citation_salience.py
import re

ROLE_PATTERNS = {
    "RELIED": [r"relied upon", r"followed", r"applied", r"held in"],
    "DISTINGUISHED": [r"distinguished", r"not applicable"],
    "OVERRULED": [r"overruled", r"set aside", r"not good law"],
}

def classify_role(context_text):
    ctx = context_text.lower()
    for role, pats in ROLE_PATTERNS.items():
        for p in pats:
            if p in ctx:
                return role
    return "MENTIONED"

def compute_salience(supporting_sentences, role):
    if not supporting_sentences:
        return 0.0
    base = sum(s["score"] for s in supporting_sentences) / len(supporting_sentences)
    weight = {"RELIED": 0.4, "DISTINGUISHED": 0.2, "MENTIONED": 0.1, "OVERRULED": -0.3}[role]
    return round(base + weight, 3)
