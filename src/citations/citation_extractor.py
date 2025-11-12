# src/citations/citation_extractor.py
import re
from src.utils import sentence_spans
from sentence_transformers import SentenceTransformer, util

CITATION_PATTERNS = [
    r"(?P<case1>[A-Z][\w\.\-\,\s&]+ v(?:s|\.|s\.)? [A-Z][\w\.\-\,\s&]+),\s*\((?P<year1>\d{4})\)\s*(?P<vol1>\d+)\s*SCC\s*(?P<page1>\d+)",
    r"\((?P<year2>\d{4})\)\s*(?P<vol2>\d+)\s*SCC\s*(?P<page2>\d+)",
    r"(?P<year3>\d{4})\s*SCC\s*(?P<page3>\d+)",
    r"(?P<AIR>AIR\s+(?P<airy>\d{4})\s*(?P<court1>SC|HIGH|ALL)\s*(?P<airpage>\d+))",
    r"(?P<MANU>MANU\/[A-Z0-9\-\/]+)",
    r"(?P<CRLA>\bCrl\.?A\.?\s*\d+\/\d{4}\b)",
    r"(?P<case_simple>[A-Z][\w\.\-\,\s&]+ v(?:s|\.|s\.)? [A-Z][\w\.\-\,\s&]+)\s*,?\s*\(?\d{4}\)?"
]
CITATION_REGEX = re.compile("|".join(CITATION_PATTERNS), flags=re.IGNORECASE)

_sbert = None
def get_sbert():
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return _sbert

def find_citations(text):
    matches=[]
    for m in CITATION_REGEX.finditer(text):
        raw = m.group(0).strip()
        gd = m.groupdict()
        year = gd.get("year1") or gd.get("year2") or gd.get("year3") or gd.get("airy")
        page = gd.get("page1") or gd.get("page2") or gd.get("page3") or gd.get("airpage")
        case = gd.get("case1") or gd.get("case_simple")
        reporter = None
        if gd.get("MANU"): reporter = gd.get("MANU")
        elif gd.get("CRLA"): reporter = gd.get("CRLA")
        elif "SCC" in raw.upper() or "AIR" in raw.upper(): reporter = "SCC" if "SCC" in raw.upper() else "AIR"
        matches.append({"match": raw, "start": m.start(), "end": m.end(), "year": year, "page": page, "case": case, "reporter": reporter})
    seen=set(); uniq=[]
    for c in matches:
        key = re.sub(r"\s+"," ", c["match"].lower())
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq

def canonicalize(c):
    if not isinstance(c, dict): return re.sub(r"\s+"," ", str(c).strip())
    parts=[]
    if c.get("case"): parts.append(re.sub(r"[^\w\s]","", c["case"]).strip().replace(" ","_")[:80])
    if c.get("year"): parts.append(str(c["year"]))
    if c.get("reporter"): parts.append(str(c["reporter"]).upper())
    if c.get("page"): parts.append(str(c["page"]))
    return "::".join(parts) if parts else c.get("match")

def build_contexts(text, citations, window=5, top_k=8):
    spans = sentence_spans(text)
    sentences = [s for (_,_,s) in spans]
    offsets = [(s,e) for s,e,_ in spans]
    sbert = get_sbert()
    sent_embs = sbert.encode(sentences, convert_to_tensor=True) if sentences else None
    contexts=[]
    for cit in citations:
        sidx = None
        for i,(st,en) in enumerate(offsets):
            if (st <= cit["start"] < en) or (st < cit["end"] <= en) or (cit["start"]<=st and cit["end"]>=en):
                sidx=i; break
        if sidx is None:
            sidx = min(range(len(offsets)), key=lambda i: abs(offsets[i][0]-cit["start"])) if offsets else 0
        start = max(0, sidx-window); end = min(len(sentences), sidx+window+1)
        ctx_window = sentences[start:end]
        supporting=[]
        if sent_embs is not None and 0 <= sidx < sent_embs.shape[0]:
            q_emb = sent_embs[sidx].unsqueeze(0)
            hits = util.semantic_search(q_emb, sent_embs, top_k=top_k+3)[0]
            for h in hits:
                st = sentences[h["corpus_id"]]
                if len(re.sub(r"\W","",st)) <= 3: continue
                supporting.append({"idx": h["corpus_id"], "sentence": st, "score": float(h["score"])})
                if len(supporting) >= top_k: break
        contexts.append({
            "citation": canonicalize(cit),
            "raw": cit["match"],
            "sent_index": sidx,
            "context_window": ctx_window,
            "supporting_sentences": supporting
        })
    return contexts
