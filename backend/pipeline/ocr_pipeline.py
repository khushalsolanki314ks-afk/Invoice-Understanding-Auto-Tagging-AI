"""
Week 2: OCR Pipeline Implementation
====================================
Converts invoice files (PDF / scanned images) into machine-readable text.
Handles common OCR issues: misread numbers, broken tables, skewed scans.

Supported input formats:
  - PDF (text-based or scanned)
  - PNG / JPG / TIFF images

Usage:
    from pipeline.ocr_pipeline import OCRPipeline
    result = OCRPipeline().process("invoice.pdf")
"""

import io
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class OCRResult:
    """Output of the OCR pipeline for a single invoice file."""
    raw_text: str                          # Full extracted text
    source_type: str                       # "pdf_text" | "pdf_scanned" | "image"
    page_count: int = 1
    confidence: float = 0.0               # 0–1 confidence score
    warnings: list[str] = field(default_factory=list)
    processing_steps: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Pre-processing helpers
# ──────────────────────────────────────────────

class ImagePreprocessor:
    """
    Cleans up scanned images before passing to Tesseract.
    Steps: grayscale → denoise → deskew → threshold
    """

    @staticmethod
    def preprocess(image):
        """
        Apply image cleanup pipeline.
        Requires: opencv-python (cv2), numpy
        """
        try:
            import cv2
            import numpy as np

            # Convert PIL image → OpenCV
            img_array = np.array(image)

            # 1. Grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # 2. Denoise (removes scanner noise)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)

            # 3. Adaptive thresholding (handles uneven lighting)
            thresh = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 4. Deskew
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.5:  # Only rotate if significant skew
                    (h, w) = thresh.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    thresh = cv2.warpAffine(
                        thresh, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )

            # Convert back to PIL
            from PIL import Image
            return Image.fromarray(thresh)

        except ImportError:
            logger.warning("opencv-python not installed — skipping image preprocessing")
            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image


# ──────────────────────────────────────────────
# OCR Pipeline
# ──────────────────────────────────────────────

class OCRPipeline:
    """
    Main OCR pipeline. Auto-detects whether input is:
      - A text-based PDF  → extract directly with pdfplumber
      - A scanned PDF     → convert pages to images → Tesseract
      - An image file     → preprocess → Tesseract
    """

    # Tesseract config for invoice documents (no restrictive whitelist to support multiple languages)
    TESSERACT_CONFIG = r"--oem 3 --psm 6"

    def __init__(self, lang: str = "eng", preprocess: bool = True):
        self.lang = lang
        self.preprocess = preprocess
        self.preprocessor = ImagePreprocessor()

    # ── Public API ──────────────────────────────

    def process(self, file_path: str) -> OCRResult:
        """
        Process an invoice file and return extracted text + metadata.

        Args:
            file_path: Path to PDF or image file

        Returns:
            OCRResult with raw_text, confidence, warnings
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Invoice file not found: {file_path}")

        ext = path.suffix.lower()
        logger.info(f"Processing invoice: {path.name} ({ext})")

        if ext == ".pdf":
            return self._process_pdf(path)
        elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
            return self._process_image(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use PDF or image files.")

    def process_bytes(self, data: bytes, filename: str) -> OCRResult:
        """Process invoice from raw bytes (e.g., API upload)."""
        import tempfile
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        logger.info(f"Wrote upload to temporary path for OCR: {tmp_path} (size={os.path.getsize(tmp_path)} bytes)")
        try:
            return self.process(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.warning(f"Failed to remove temporary file: {tmp_path}")

    # ── PDF Processing ──────────────────────────

    def _process_pdf(self, path: Path) -> OCRResult:
        """Detect if PDF is text-based or scanned, then extract accordingly."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        steps = []
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"

        # Heuristic: if we got meaningful text, it's a text PDF
        word_count = len(full_text.split())
        if word_count > 20:
            steps.append("pdfplumber_text_extraction")
            text = self._clean_text(full_text)
            confidence = self._estimate_confidence(text)
            logger.info(f"Text PDF detected — extracted {word_count} words")
            return OCRResult(
                raw_text=text,
                source_type="pdf_text",
                page_count=page_count,
                confidence=confidence,
                processing_steps=steps
            )
        else:
            # Scanned PDF — convert to images and run Tesseract
            logger.info("Scanned PDF detected — running image OCR on pages")
            return self._process_scanned_pdf(path, page_count)

    def _process_scanned_pdf(self, path: Path, page_count: int) -> OCRResult:
        """Convert scanned PDF pages to images and OCR each page."""
        steps = ["pdf_to_image_conversion"]
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("Install pdf2image: pip install pdf2image")

        pages = convert_from_path(str(path), dpi=300)
        all_text = []
        confidences = []

        for i, page_img in enumerate(pages):
            logger.debug(f"OCR processing page {i+1}/{len(pages)}")
            if self.preprocess:
                page_img = self.preprocessor.preprocess(page_img)
                steps.append(f"image_preprocess_p{i+1}")

            text, conf = self._tesseract_ocr(page_img)
            all_text.append(text)
            confidences.append(conf)

        combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
        cleaned = self._clean_text(combined_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            raw_text=cleaned,
            source_type="pdf_scanned",
            page_count=page_count,
            confidence=avg_conf,
            processing_steps=steps,
            warnings=self._check_quality(cleaned)
        )

    # ── Image Processing ────────────────────────

    def _process_image(self, path: Path) -> OCRResult:
        """OCR a single image file."""
        steps = []
        try:
            from PIL import Image
            img = Image.open(path)
        except ImportError:
            raise ImportError("Install Pillow: pip install Pillow")

        if self.preprocess:
            img = self.preprocessor.preprocess(img)
            steps.append("image_preprocessing")

        text, confidence = self._tesseract_ocr(img)
        cleaned = self._clean_text(text)
        steps.append("tesseract_ocr")

        return OCRResult(
            raw_text=cleaned,
            source_type="image",
            page_count=1,
            confidence=confidence,
            processing_steps=steps,
            warnings=self._check_quality(cleaned)
        )

    # ── Core OCR ───────────────────────────────

    def _tesseract_ocr(self, image) -> tuple[str, float]:
        """Run Tesseract OCR on a PIL image. Returns (text, confidence)."""
        try:
            import pytesseract
            # Get detailed output with confidence scores
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.TESSERACT_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
            # Filter out low-confidence / empty tokens
            words = []
            confs = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 0 and word.strip():
                    words.append(word)
                    confs.append(conf)

            text = " ".join(words)
            avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0
            return text, avg_conf

        except ImportError:
            raise ImportError(
                "Install pytesseract + Tesseract:\n"
                "  pip install pytesseract\n"
                "  brew install tesseract  (macOS)\n"
                "  sudo apt install tesseract-ocr  (Ubuntu)"
            )

    # ── Post-processing ─────────────────────────

    def _clean_text(self, text: str) -> str:
        """
        Fix common OCR artifacts:
        - Replace 0 → O confusions in known contexts
        - Normalize whitespace
        - Fix common number misreads
        """
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix common OCR number/letter confusions
        # "lnvoice" → "Invoice"
        text = re.sub(r"\blnvoice\b", "Invoice", text)
        # "O" in numeric context → "0"
        text = re.sub(r"(?<=\d)O(?=\d)", "0", text)
        # Fix rupee symbol variants
        text = re.sub(r"[₹Rs\.]+\s*", "₹", text)

        return text.strip()

    def _estimate_confidence(self, text: str) -> float:
        """
        Heuristic confidence for text-PDF extraction.
        Checks presence of key invoice keywords.
        """
        keywords = ["invoice", "date", "amount", "total", "buyer", "seller", "gst", "due"]
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw in text_lower)
        return min(found / len(keywords) + 0.5, 1.0)

    def _check_quality(self, text: str) -> list[str]:
        """Return warnings if OCR quality seems low."""
        warnings = []
        words = text.split()
        if len(words) < 10:
            warnings.append("Very little text extracted — possible blank/corrupted scan")
        if len([w for w in words if len(w) == 1]) > len(words) * 0.3:
            warnings.append("High ratio of single characters — OCR quality may be poor")
        return warnings


# ──────────────────────────────────────────────
# Mock OCR for Development (no Tesseract needed)
# ──────────────────────────────────────────────

class MockOCRPipeline:
    """
    Drop-in replacement for OCRPipeline during development.
    Returns realistic mock OCR text without needing Tesseract installed.
    """

    MOCK_INVOICE_TEXT = """
    INVOICE

    Invoice No: BM-2024-09001
    Invoice Date: 01/09/2024
    Due Date: 01/10/2024

    Seller: Apex Packaging Co.
    Address: 45 Industrial Estate, Pune - 411001
    GSTIN: 27AAPCA1234F1ZX

    Bill To:
    Reliance Retail Ltd
    Navi Mumbai, Maharashtra - 400701

    ─────────────────────────────────────────────
    Description               Qty    Rate    Amount
    ─────────────────────────────────────────────
    HDPE Bags 50kg             500   650.00  3,25,000
    Corrugated Boxes L         200   767.50  1,53,500
    ─────────────────────────────────────────────
    Subtotal                                4,78,500
    GST @ 18%                                86,130
    TOTAL                                  5,64,630
    ─────────────────────────────────────────────

    Bank: HDFC Bank | A/C: 1234567890 | IFSC: HDFC0001234
    Payment Terms: Net 30 days
    """

    def process(self, file_path: str) -> OCRResult:
        logger.info(f"[MOCK] OCR processing: {file_path}")
        return OCRResult(
            raw_text=self.MOCK_INVOICE_TEXT.strip(),
            source_type="mock",
            page_count=1,
            confidence=0.95,
            processing_steps=["mock_extraction"],
            warnings=[]
        )

    def process_bytes(self, data: bytes, filename: str) -> OCRResult:
        return self.process(filename)
