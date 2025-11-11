# ⚖️ Citation-Aware Bilingual Legal Document Summarizer

📚 *A research-driven pipeline for legal document understanding and bilingual summarization using mT5, Legal-BERT, and citation-aware reasoning.*

---

## 🚀 Overview
This project presents a **Citation-Aware Legal Summarization System** that processes Indian court judgments (English or Hindi) into **bilingual summaries (English + Hindi)** while **preserving legal citations and contextual reasoning**.

The model is fine-tuned on a curated dataset of **40 Indian court judgments** collected from the **[Indian Kanoon](https://indiankanoon.org)** database, focusing on **citation-based contextual summarization** and **parallel translation**.

---

## 🧠 Pipeline Architecture


 1️⃣ Input Legal Document (English/Hindi)
        │
        ▼
2️⃣ Language Detection & Translation (IndicTrans2)
        │
        ▼
3️⃣ Legal NER & Citation Extraction (Legal-BERT + Regex)
        │
        ▼
4️⃣ Citation-Aware Context Builder (Novel Layer)
   → Adds key legal citation context
   → Highlights relevant paragraphs
        │
        ▼
5️⃣ Citation-Guided Summarization (Novel Layer)
   → mT5 fine-tuned on Indian legal corpus
   → Weighted attention on citation sentences
        │
        ▼
6️⃣ Bilingual Alignment Translator
   → Generates aligned English + Hindi summaries
        │
        ▼
7️⃣ Output Layer
   → JSON: {English_summary, Hindi_summary, citations, sections}


##Dataset

Source: 40 Indian High Court and Supreme Court judgments collected from Indian Kanoon

Languages: English, Hindi

Annotations: Citation contexts (citation_contexts.jsonl)

Format Example:

{
  "doc_id": "Alemla_Jamir_vs_NIA_2025",
  "citation": "Alemla Jamir vs NIA",
  "context_window": ["... paragraphs around citation ..."],
  "supporting_sentences": ["... relevant lines ..."],
  "role": "MENTIONED",
  "salience": 0.92,
}

###Preprocessing:

Noise removal (headnotes, signatures, formatting)

Sentence segmentation

Citation normalization

Salience tagging via cosine similarity + Legal-BERT

##Output Example
{
  "doc_id": "Suresh_Kalmadi_vs_CBI_2012",
  "summary_en_ctxaware": "The Court held that the object of bail is to ensure presence at trial, not punishment. Bail is the rule, jail the exception.",
  "summary_hi_ctxaware": "न्यायालय ने कहा कि जमानत का उद्देश्य अभियुक्त की उपस्थिति सुनिश्चित करना है, दंड नहीं। जमानत नियम है, जेल अपवाद।",
  "citation_contexts": [
    {
      "citation": "Sanjay Chandra v. CBI (2011)",
      "role": "RELIED",
      "salience": 0.88,
      "context_window": ["The Court relied on Sanjay Chandra v. CBI..."]
    }
  ]
}
