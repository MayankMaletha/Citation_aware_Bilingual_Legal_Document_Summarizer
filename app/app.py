# app/app.py
import os, sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import streamlit as st
from io import BytesIO
import json

from main import process_pdf_file, save_json  # ✅ clean import

st.set_page_config(page_title="Legal Summarizer (PDF → EN/HI + Citations)", layout="wide")

st.title("📄 Legal Summarizer – PDF → English/Hindi Summary + Citations")
st.caption("Drop a court judgment PDF. I’ll extract text, detect language, find citations, and produce bilingual summaries.")

with st.sidebar:
    st.header("Settings")
    ocr = st.toggle("Use OCR (for scanned PDFs)", value=False)
    ocr_page_limit = st.number_input("OCR page limit (optional)", min_value=1, value=10, step=1)
    salience_threshold = st.slider("Salience threshold", 0.0, 1.0, 0.55, 0.01)
    max_contexts = st.slider("Max contexts in prompt", 1, 16, 8, 1)
    output_dir = st.text_input("Save JSON outputs to folder", value="output_folder/json")
    st.divider()
    st.markdown("**Tip:** Run from repo root:**\n```bash\nstreamlit run app/app.py\n```")

uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded:
    for up in uploaded:
        st.subheader(f"File: {up.name}")
        with st.spinner("Processing… please wait"):
            tmp_dir = Path(".streamlit_tmp"); tmp_dir.mkdir(exist_ok=True)
            pdf_path = tmp_dir / up.name
            pdf_path.write_bytes(up.read())

            try:
                result = process_pdf_file(
                    str(pdf_path),
                    ocr=ocr,
                    ocr_page_limit=int(ocr_page_limit),
                    salience_threshold=float(salience_threshold),
                    max_contexts=int(max_contexts),
                    translate_to_hi=True
                )
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                continue

        # Show summaries
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🇬🇧 English Summary")
            st.write(result.get("summary_en_ctxaware", ""))
        with col2:
            st.markdown("### 🇮🇳 सारांश (Hindi Summary)")
            st.write(result.get("summary_hi_ctxaware", ""))

        # Show citation contexts
        st.markdown("### 📚 Citation Contexts")
        ctxs = result.get("citation_contexts", [])
        if not ctxs:
            st.info("No citations detected.")
        else:
            import pandas as pd
            rows = [{
                "citation": c.get("citation") or c.get("raw"),
                "role": c.get("role", ""),
                "salience": round(float(c.get("salience", 0.0)), 3),
                "context_window": " ".join(c.get("context_window", [])[:3])
            } for c in ctxs]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

        # Download JSON
        out_bytes = BytesIO(json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button(
            label="⬇️ Download JSON",
            data=out_bytes,
            file_name=f"{result['doc_id']}.json",
            mime="application/json"
        )

        if output_dir:
            try:
                saved = save_json(result, output_dir)
                st.success(f"✅ Saved to {saved}")
            except Exception as e:
                st.warning(f"⚠️ Could not save: {e}")
