# evaluation/extract_citations_from_summary.py
import re

PATTERNS = [
    r"[A-Z][\w\.\-\,\s&]+ v(?:s|\.|s\.)? [A-Z][\w\.\-\,\s&]+(?:\s*\(\d{4}\))?",
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+",
    r"\d{4}\s*SCC\s*\d+",
    r"AIR\s+\d{4}\s*(?:SC|HIGH|ALL)\s*\d+",
    r"MANU\/[A-Z0-9\-\/]+",
    r"\bCrl\.?A\.?\s*\d+\/\d{4}\b",
]
RX = re.compile("|".join(PATTERNS), flags=re.IGNORECASE)

def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())

def extract(summary_text: str):
    found = set()
    for m in RX.finditer(summary_text or ""):
        cand = m.group(0)
        # tiny cleanups
        cand = cand.strip(" ,.;:()[]")
        found.add(_norm(cand))
    return found
