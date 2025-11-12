# src/cleaning/cleaner.py
import re
_HEADER_PATTERNS = [
    r"Indian Kanoon\s*-?\s*http[s]?:\/\/\S+",
    r"^\s*\d+\s*$",
    r"^\f",
]
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")

def clean_text(raw_text):
    if not raw_text:
        return "", ""
    t = re.sub(r"\r\n?", "\n", raw_text)
    lines = t.split("\n")
    out_lines = []
    for ln in lines:
        ln2 = ln.strip().replace("\f"," ").strip()
        if not ln2:
            out_lines.append("")  # keep paragraph break
            continue
        skip=False
        for pat in _HEADER_PATTERNS:
            if re.search(pat, ln2, flags=re.IGNORECASE):
                skip=True; break
        if not skip:
            out_lines.append(ln2)
    text_preserve = "\n".join(out_lines)
    text_preserve = _HYPHEN_BREAK_RE.sub(r"\1\2", text_preserve)
    text_preserve = re.sub(r"[ \t]{2,}", " ", text_preserve)
    text_preserve = re.sub(r"\n{3,}", "\n\n", text_preserve).strip()
    text_single = re.sub(r"\s+", " ", text_preserve).strip()
    return text_preserve, text_single

def clean_for_json(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()
