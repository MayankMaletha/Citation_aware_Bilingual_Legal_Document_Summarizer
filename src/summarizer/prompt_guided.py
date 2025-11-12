# src/summarizer/prompt_guided.py

def build_guided_prompt(working_text, contexts, max_contexts=8):
    # pick highest-salience contexts first
    ctxs = sorted(contexts, key=lambda c: c.get("salience", 0), reverse=True)[:max_contexts]
    citations = [(c.get("raw") or "").strip() for c in ctxs if c.get("raw")]
    citations = [c for c in citations if c]

    bullet_list = "\n".join(f"- {c}" for c in citations)

    return f"""
### TASK ###
Summarize the legal reasoning and decision clearly.

### REQUIREMENT ###
If any of the following citations influenced the judgment,
**include the exact citation text in the summary** (copy them verbatim, do not rephrase).

### IMPORTANT CITATIONS ###
{bullet_list}

### DOCUMENT CONTEXT ###
{working_text[:8000]}
""".strip()
