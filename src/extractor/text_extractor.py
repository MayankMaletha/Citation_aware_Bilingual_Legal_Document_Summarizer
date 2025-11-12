# src/extractor/text_extractor.py
from pdfminer.high_level import extract_text
from pathlib import Path
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_IMPORTED = True
except Exception:
    OCR_IMPORTED = False
from src.utils import md5_of_file
import logging
logger = logging.getLogger(__name__)

def extract_text_pdfminer(pdf_path):
    try:
        return extract_text(str(pdf_path)) or ""
    except Exception as e:
        logger.warning("pdfminer failed: %s", e)
        return ""

def ocr_pdf(pdf_path, dpi=250, page_limit=None):
    if not OCR_IMPORTED:
        return ""
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if page_limit:
        pages = pages[:page_limit]
    txts = []
    for p in pages:
        txts.append(pytesseract.image_to_string(p))
    return "\n".join(txts)

def extract(pdf_path, ocr=False, ocr_page_limit=None):
    pdf_path = Path(pdf_path)
    text = extract_text_pdfminer(pdf_path)
    used_ocr = False
    if not text or len(text.strip()) < 300:
        if ocr and OCR_IMPORTED:
            text = ocr_pdf(pdf_path, page_limit=ocr_page_limit)
            used_ocr = True
    return {"text": text or "", "ocr_used": used_ocr, "md5": md5_of_file(pdf_path)}
