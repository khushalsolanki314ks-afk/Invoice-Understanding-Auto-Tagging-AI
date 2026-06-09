"""
Week 4: NLP-Based Field Extraction
====================================
Extracts structured invoice fields from raw OCR text using:
  1. Rule-based extraction (regex, keyword proximity)
  2. spaCy NLP for named entity recognition (names, orgs, dates)

Output: InvoiceFields dataclass with confidence scores per field.

Usage:
    from pipeline.field_extractor import FieldExtractor
    fields = FieldExtractor().extract(structured_invoice, raw_text)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from dateutil import parser as date_parser
from loguru import logger


# ──────────────────────────────────────────────
# Output Data Class
# ──────────────────────────────────────────────

@dataclass
class ExtractedField:
    """A single extracted field with its value and confidence."""
    value: Optional[str]
    confidence: float       # 0.0 – 1.0
    method: str             # "regex" | "nlp" | "heuristic" | "none"


@dataclass
class InvoiceFields:
    """
    All extracted invoice fields with individual confidence scores.
    This is the main output of the Field Extractor.
    """
    invoice_number: ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    invoice_date:   ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    due_date:       ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    buyer_name:     ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    seller_name:    ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    amount:         ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    gst_amount:     ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    total_amount:   ExtractedField = field(default_factory=lambda: ExtractedField(None, 0.0, "none"))
    currency:       ExtractedField = field(default_factory=lambda: ExtractedField("INR", 0.8, "heuristic"))

    # Parsed/normalized versions (for downstream use)
    invoice_date_parsed: Optional[datetime] = None
    due_date_parsed:     Optional[datetime] = None
    amount_float:        Optional[float]    = None
    total_amount_float:  Optional[float]    = None

    # Overall extraction quality
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        def field_dict(f: ExtractedField) -> dict:
            return {"value": f.value, "confidence": round(f.confidence, 2), "method": f.method}

        return {
            "invoice_number": field_dict(self.invoice_number),
            "invoice_date":   field_dict(self.invoice_date),
            "due_date":       field_dict(self.due_date),
            "buyer_name":     field_dict(self.buyer_name),
            "seller_name":    field_dict(self.seller_name),
            "amount":         field_dict(self.amount),
            "gst_amount":     field_dict(self.gst_amount),
            "total_amount":   field_dict(self.total_amount),
            "currency":       field_dict(self.currency),
            "parsed": {
                "invoice_date": self.invoice_date_parsed.isoformat() if self.invoice_date_parsed else None,
                "due_date": self.due_date_parsed.isoformat() if self.due_date_parsed else None,
                "amount_float": self.amount_float,
                "total_amount_float": self.total_amount_float,
            },
            "overall_confidence": round(self.overall_confidence, 2),
        }


# ──────────────────────────────────────────────
# Regex Patterns
# ──────────────────────────────────────────────

# Invoice number: alphanumeric with dashes/slashes
INVOICE_NO_PATTERNS = [
    r"(?:invoice|inv|bill)\s*(?:no|number|#|num)?\.?\s*[:\-]?\s*([A-Z]{2,}-\d{4}-\d{4,})",
    r"(?:invoice|inv|bill)\s*(?:no|number|#|num)?\.?\s*[:\-]?\s*([A-Z0-9]{3,}[\/\-][A-Z0-9\-\/]{3,})",
    r"(?:invoice|inv)\s*[:#\-]?\s*([A-Z0-9\-\/]{5,20})",
]

# Date formats: DD/MM/YYYY, MM-DD-YYYY, DD Month YYYY, YYYY-MM-DD
DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b",
    r"\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
]

# Amount: ₹, Rs, INR followed by digits
AMOUNT_PATTERNS = [
    r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{2})?)",
    r"([\d,]+(?:\.\d{2})?)\s*(?:₹|Rs\.?|INR)",
    r"\b([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?)\b",
]

# GST patterns
GST_PATTERNS = [
    r"(?:gst|cgst|sgst|igst)\s*(?:@\s*[\d\.]+%)?\s*[:\-]?\s*(?:₹|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)",
]

# Currency detection
CURRENCY_MAP = {
    r"₹|Rs\.?|INR": "INR",
    r"\$|USD": "USD",
    r"€|EUR": "EUR",
    r"£|GBP": "GBP",
}

# ──────────────────────────────────────────────
# Field Extractor
# ──────────────────────────────────────────────

class FieldExtractor:
    """
    Extracts invoice fields using rule-based (regex) and NLP (spaCy) methods.
    Falls back gracefully if spaCy is not installed.
    """

    def __init__(self, use_nlp: bool = True):
        self.use_nlp = use_nlp
        self._nlp = None
        if use_nlp:
            self._load_nlp()

    def _load_nlp(self):
        """Lazy-load spaCy model."""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
        except ImportError:
            logger.warning("spaCy not installed — using regex only")
            self.use_nlp = False
        except OSError:
            logger.warning("spaCy model not found — run: python -m spacy download en_core_web_sm")
            self.use_nlp = False

    # ── Main Entry ──────────────────────────────

    def extract(self, raw_text: str, structured=None) -> InvoiceFields:
        """
        Extract all invoice fields from raw OCR text.
        Optionally uses pre-structured data (from TextStructurer) to improve accuracy.

        Args:
            raw_text: Full OCR text
            structured: Optional StructuredInvoice from TextStructurer

        Returns:
            InvoiceFields with values and confidence scores
        """
        fields = InvoiceFields()
        text = raw_text

        # ── 1. Invoice Number ──
        # Prefer structured header, fall back to full-text regex
        inv_no_hint = structured.invoice_number_raw if structured else None
        fields.invoice_number = self._extract_invoice_number(text, inv_no_hint)

        # ── 2. Dates ──
        inv_date_hint = structured.invoice_date_raw if structured else None
        due_date_hint = structured.due_date_raw if structured else None
        fields.invoice_date, fields.due_date = self._extract_dates(text, inv_date_hint, due_date_hint)
        fields.invoice_date_parsed = self._parse_date(fields.invoice_date.value)
        fields.due_date_parsed = self._parse_date(fields.due_date.value)

        # ── 3. Entity Names (buyer/seller) ──
        buyer_hint = structured.buyer_block if structured else None
        seller_hint = structured.seller_block if structured else None
        fields.buyer_name, fields.seller_name = self._extract_entities(text, buyer_hint, seller_hint)

        # ── 4. Amounts ──
        total_hint = structured.total_raw if structured else None
        subtotal_hint = structured.subtotal_raw if structured else None
        tax_hint = structured.tax_raw if structured else None
        fields.amount, fields.gst_amount, fields.total_amount = self._extract_amounts(
            text, subtotal_hint, tax_hint, total_hint
        )
        fields.amount_float = self._to_float(fields.amount.value)
        fields.total_amount_float = self._to_float(fields.total_amount.value)

        # ── 5. Currency ──
        fields.currency = self._detect_currency(text)

        # ── Overall confidence ──
        fields.overall_confidence = self._compute_overall_confidence(fields)

        logger.info(
            f"Extraction complete — confidence: {fields.overall_confidence:.2f} | "
            f"invoice: {fields.invoice_number.value} | buyer: {fields.buyer_name.value} | "
            f"total: {fields.total_amount.value}"
        )
        return fields

    # ── Invoice Number ──────────────────────────

    def _extract_invoice_number(self, text: str, hint: str = None) -> ExtractedField:
        # Use hint from structured parser if available and looks valid
        if hint and re.match(r"[A-Z0-9\-\/]{4,}", hint):
            return ExtractedField(hint, 0.9, "structured_parser")

        # Try regex patterns
        for pat in INVOICE_NO_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                value = m.group(1).strip().upper()
                return ExtractedField(value, 0.85, "regex")

        return ExtractedField(None, 0.0, "none")

    # ── Dates ───────────────────────────────────

    def _extract_dates(
        self, text: str, inv_hint: str = None, due_hint: str = None
    ) -> tuple[ExtractedField, ExtractedField]:
        """Extract invoice date and due date."""
        # Find all date occurrences in text with their positions
        all_dates = []
        for pat in DATE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                all_dates.append((m.start(), m.group(1)))

        all_dates.sort(key=lambda x: x[0])  # Sort by position

        invoice_date = ExtractedField(None, 0.0, "none")
        due_date = ExtractedField(None, 0.0, "none")

        # Use hints first
        if inv_hint:
            invoice_date = ExtractedField(inv_hint, 0.85, "structured_parser")
        if due_hint:
            due_date = ExtractedField(due_hint, 0.85, "structured_parser")

        # Labeled date extraction (higher confidence)
        inv_labeled = re.search(
            r"(?:invoice\s*date|date\s*of\s*invoice|dated?)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})",
            text, re.IGNORECASE
        )
        if inv_labeled:
            invoice_date = ExtractedField(inv_labeled.group(1), 0.92, "regex_labeled")

        due_labeled = re.search(
            r"(?:due\s*date|payment\s*date|pay\s*by|due\s*on)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})",
            text, re.IGNORECASE
        )
        if due_labeled:
            due_date = ExtractedField(due_labeled.group(1), 0.92, "regex_labeled")

        # Fall back to positional dates if still missing
        if not invoice_date.value and all_dates:
            invoice_date = ExtractedField(all_dates[0][1], 0.5, "positional")
        if not due_date.value and len(all_dates) >= 2:
            due_date = ExtractedField(all_dates[1][1], 0.5, "positional")

        return invoice_date, due_date

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        if not date_str:
            return None
        try:
            return date_parser.parse(date_str, dayfirst=True)
        except Exception:
            return None

    # ── Entity Names (Buyer / Seller) ───────────

    def _extract_entities(
        self, text: str, buyer_hint: str = None, seller_hint: str = None
    ) -> tuple[ExtractedField, ExtractedField]:
        """Extract buyer and seller names using NLP + regex."""
        buyer = ExtractedField(None, 0.0, "none")
        seller = ExtractedField(None, 0.0, "none")

        # Use structured hints (from TextStructurer) first
        if buyer_hint:
            name = self._clean_entity_name(buyer_hint)
            if name:
                buyer = ExtractedField(name, 0.75, "structured_parser")

        if seller_hint:
            name = self._clean_entity_name(seller_hint)
            if name:
                seller = ExtractedField(name, 0.75, "structured_parser")

        # Try labeled extraction with regex
        bill_to = re.search(
            r"(?:bill\s*to|buyer|sold\s*to|ship\s*to)\s*[:\-]?\s*\n?\s*([A-Z][A-Za-z\s\.\,&]+(?:Ltd|Pvt|Inc|Corp|Co\.?|LLP|LLC|Retail|Products|Technologies)?)",
            text, re.IGNORECASE
        )
        if bill_to and not buyer.value:
            buyer = ExtractedField(bill_to.group(1).strip(), 0.85, "regex_labeled")

        from_match = re.search(
            r"(?:from|seller|vendor|supplier)\s*[:\-]?\s*\n?\s*([A-Z][A-Za-z\s\.\,&]+(?:Ltd|Pvt|Inc|Corp|Co\.?|LLP|LLC|Industries|Solutions)?)",
            text, re.IGNORECASE
        )
        if from_match and not seller.value:
            seller = ExtractedField(from_match.group(1).strip(), 0.85, "regex_labeled")

        # Use spaCy NLP as fallback for org names
        if self._nlp and (not buyer.value or not seller.value):
            buyer, seller = self._nlp_extract_entities(text, buyer, seller)

        return buyer, seller

    def _nlp_extract_entities(
        self, text: str,
        buyer: ExtractedField, seller: ExtractedField
    ) -> tuple[ExtractedField, ExtractedField]:
        """Use spaCy to find ORG entities as buyer/seller fallback."""
        doc = self._nlp(text[:2000])  # Limit to first 2000 chars
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

        if orgs and not buyer.value:
            buyer = ExtractedField(orgs[0], 0.6, "nlp_ner")
        if len(orgs) >= 2 and not seller.value:
            seller = ExtractedField(orgs[1], 0.6, "nlp_ner")

        return buyer, seller

    def _clean_entity_name(self, raw: str) -> Optional[str]:
        """Clean up a raw entity name from OCR."""
        if not raw:
            return None
        # Take just the first line
        name = raw.split("\n")[0].strip()
        # Remove leading/trailing punctuation
        name = re.sub(r"^[:\-\s]+|[:\-\s,]+$", "", name)
        if len(name) < 3:
            return None
        return name[:100]

    # ── Amounts ─────────────────────────────────

    def _extract_amounts(
        self, text: str,
        subtotal_hint: str = None,
        tax_hint: str = None,
        total_hint: str = None,
    ) -> tuple[ExtractedField, ExtractedField, ExtractedField]:
        """Extract invoice amount, GST, and grand total."""
        # Use structured hints first
        amount_field = ExtractedField(None, 0.0, "none")
        gst_field = ExtractedField(None, 0.0, "none")
        total_field = ExtractedField(None, 0.0, "none")

        if subtotal_hint:
            amount_field = ExtractedField(subtotal_hint, 0.8, "structured_parser")
        if tax_hint:
            gst_field = ExtractedField(tax_hint, 0.8, "structured_parser")
        if total_hint:
            total_field = ExtractedField(total_hint, 0.8, "structured_parser")

        # Regex extraction for totals section
        total_match = re.search(
            r"(?:grand\s*total|total\s*amount|total\s*due|balance\s*due|amount\s*due)"
            r"\s*[:\-]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)",
            text, re.IGNORECASE
        )
        if total_match and not total_field.value:
            total_field = ExtractedField(total_match.group(1).replace(",", ""), 0.9, "regex_labeled")

        subtotal_match = re.search(
            r"(?:subtotal|sub\s*total|taxable\s*amount|net\s*amount)"
            r"\s*[:\-]?\s*(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d{2})?)",
            text, re.IGNORECASE
        )
        if subtotal_match and not amount_field.value:
            amount_field = ExtractedField(subtotal_match.group(1).replace(",", ""), 0.9, "regex_labeled")

        gst_match = re.search(GST_PATTERNS[0], text, re.IGNORECASE)
        if gst_match and not gst_field.value:
            gst_field = ExtractedField(gst_match.group(1).replace(",", ""), 0.88, "regex_labeled")

        # Last resort: grab the largest number in the document as total
        if not total_field.value:
            # Match long digit sequences, with optional commas/decimals
            all_amounts = re.findall(r"\b\d[\d,\.]*\b", text)
            parsed = []
            for a in all_amounts:
                try:
                    parsed.append(float(a.replace(",", "")))
                except ValueError:
                    pass
            if parsed:
                max_amount = max(parsed)
                if max_amount > 100:
                    total_field = ExtractedField(str(int(max_amount) if max_amount.is_integer() else max_amount), 0.4, "heuristic_max")

        return amount_field, gst_field, total_field

    # ── Currency ────────────────────────────────

    def _detect_currency(self, text: str) -> ExtractedField:
        for pattern, currency in CURRENCY_MAP.items():
            if re.search(pattern, text):
                return ExtractedField(currency, 0.95, "regex")
        return ExtractedField("INR", 0.5, "default")  # Default for India-focused platform

    # ── Helpers ─────────────────────────────────

    def _to_float(self, value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        s = str(value).strip()
        # Remove currency symbols and whitespace
        s = re.sub(r"[₹$€£\s]", "", s)

        # If both comma and dot present, assume comma is thousand separator (e.g., 1,234.56)
        if "," in s and "." in s:
            s = s.replace(",", "")
        # If only comma present and groups of three after comma, assume comma thousands (1,234)
        elif "," in s and re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$", s):
            s = s.replace(",", "")
        # If only comma present and it looks like decimal separator (e.g., 1234,56), convert to dot
        elif "," in s and re.match(r"^\d+,\d{1,2}$", s):
            s = s.replace(",", ".")

        try:
            return float(s)
        except ValueError:
            # Last resort: strip non-digit except dot and minus
            cleaned = re.sub(r"[^0-9\.\-]", "", s)
            try:
                return float(cleaned) if cleaned else None
            except Exception:
                return None

    def _compute_overall_confidence(self, fields: InvoiceFields) -> float:
        """
        Weighted average confidence across key fields.
        invoice_number and total_amount are most important.
        """
        weights = {
            "invoice_number": 0.25,
            "invoice_date": 0.15,
            "buyer_name": 0.25,
            "total_amount": 0.25,
            "due_date": 0.10,
        }
        total = 0.0
        for attr, weight in weights.items():
            f: ExtractedField = getattr(fields, attr)
            total += f.confidence * weight
        return total
