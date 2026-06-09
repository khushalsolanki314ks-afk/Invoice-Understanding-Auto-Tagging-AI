"""
Week 3: Text Structuring & Layout Understanding
================================================
Converts raw OCR text into structured JSON blocks:
  - Header section (invoice metadata)
  - Line items table (goods/services)
  - Totals section (subtotal, tax, grand total)

Strategy:
  - Keyword-based section detection
  - Proximity heuristics (e.g., "Total" is near the bottom)
  - Regex patterns for common invoice layouts

Usage:
    from pipeline.text_structurer import TextStructurer
    blocks = TextStructurer().structure(ocr_text)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class LineItem:
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


@dataclass
class StructuredInvoice:
    """Normalized structured representation of an invoice."""
    # Header
    raw_header: str = ""
    raw_line_items: str = ""
    raw_totals: str = ""

    # Extracted from header
    invoice_number_raw: str = ""
    invoice_date_raw: str = ""
    due_date_raw: str = ""
    seller_block: str = ""
    buyer_block: str = ""

    # Line items
    line_items: list[LineItem] = field(default_factory=list)

    # Totals
    subtotal_raw: str = ""
    tax_raw: str = ""
    total_raw: str = ""

    # Metadata
    layout_confidence: float = 0.0
    section_boundaries: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Section Detection Keywords
# ──────────────────────────────────────────────

HEADER_KEYWORDS = [
    r"invoice\s*(no|number|#|num)?",
    r"inv\.?\s*no",
    r"bill\s*(to|from)",
    r"seller|vendor|from",
    r"buyer|client|to",
    r"date",
    r"gstin",
    r"pan\s*no",
]

LINE_ITEM_KEYWORDS = [
    r"description",
    r"particulars",
    r"item",
    r"qty|quantity",
    r"rate|unit\s*price",
    r"amount|value",
    r"s\.?\s*no",
]

TOTALS_KEYWORDS = [
    r"subtotal|sub\s*total",
    r"total\s*amount",
    r"grand\s*total",
    r"gst|cgst|sgst|igst",
    r"tax",
    r"net\s*amount",
    r"balance\s*due",
]

# ──────────────────────────────────────────────
# Text Structurer
# ──────────────────────────────────────────────

class TextStructurer:
    """
    Segments raw OCR text into header / line items / totals sections.
    Works with multiple invoice layouts using keyword + position heuristics.
    """

    def structure(self, raw_text: str) -> StructuredInvoice:
        """
        Main entry point. Takes raw OCR text, returns StructuredInvoice.
        """
        if not raw_text or len(raw_text.strip()) < 20:
            logger.warning("Empty or very short OCR text received")
            return StructuredInvoice()

        lines = self._normalize_lines(raw_text)
        logger.info(f"Structuring {len(lines)} lines of OCR text")

        # Detect section boundaries
        header_end, items_start, items_end, totals_start = self._detect_sections(lines)

        result = StructuredInvoice()
        result.section_boundaries = {
            "header_end": header_end,
            "items_start": items_start,
            "items_end": items_end,
            "totals_start": totals_start,
        }

        # Split into sections
        header_lines = lines[:header_end]
        item_lines = lines[items_start:items_end] if items_start < items_end else []
        total_lines = lines[totals_start:] if totals_start < len(lines) else []

        result.raw_header = "\n".join(header_lines)
        result.raw_line_items = "\n".join(item_lines)
        result.raw_totals = "\n".join(total_lines)

        # Parse each section
        self._parse_header(result, header_lines)
        self._parse_line_items(result, item_lines)
        self._parse_totals(result, total_lines)

        result.layout_confidence = self._score_layout(result)
        logger.info(f"Structure complete — confidence: {result.layout_confidence:.2f}")

        return result

    # ── Section Detection ───────────────────────

    def _detect_sections(self, lines: list[str]) -> tuple[int, int, int, int]:
        """
        Returns (header_end, items_start, items_end, totals_start) line indices.
        Uses keyword scoring to find section transitions.
        """
        n = len(lines)
        header_end = max(1, n // 4)    # Default: first 25% is header
        items_start = header_end
        items_end = max(items_start + 1, n * 3 // 4)
        totals_start = items_end

        # Score each line for which section it likely belongs to
        header_scores = []
        item_scores = []
        total_scores = []

        for line in lines:
            ll = line.lower()
            header_scores.append(self._score_keywords(ll, HEADER_KEYWORDS))
            item_scores.append(self._score_keywords(ll, LINE_ITEM_KEYWORDS))
            total_scores.append(self._score_keywords(ll, TOTALS_KEYWORDS))

        # Find the line with the highest line-item table header score
        table_header_line = self._find_table_header(lines)
        if table_header_line is not None:
            items_start = table_header_line
            header_end = table_header_line

        # Find where totals section begins
        for i in range(n - 1, -1, -1):
            if total_scores[i] > 0 and i > items_start:
                totals_start = i
                items_end = i
                break

        return header_end, items_start, items_end, totals_start

    def _find_table_header(self, lines: list[str]) -> Optional[int]:
        """
        Find the line that looks like an items table header:
        e.g., "Description | Qty | Rate | Amount"
        """
        required_cols = [r"desc|particular|item", r"qty|quantity", r"amount|total"]
        for i, line in enumerate(lines):
            ll = line.lower()
            matches = sum(1 for pat in required_cols if re.search(pat, ll))
            if matches >= 2:
                return i
        return None

    def _score_keywords(self, text: str, patterns: list[str]) -> int:
        return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    # ── Header Parsing ──────────────────────────

    def _parse_header(self, result: StructuredInvoice, lines: list[str]) -> None:
        """Extract invoice number, dates, seller/buyer from header lines."""
        text = "\n".join(lines)

        # Invoice number
        inv_match = re.search(
            r"(?:invoice|inv|bill)\s*(?:no|number|#|num)?\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})",
            text, re.IGNORECASE
        )
        if inv_match:
            result.invoice_number_raw = inv_match.group(1).strip()

        # Invoice date (various formats)
        date_patterns = [
            r"invoice\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"dated?\s*[:\-]?\s*(\d{1,2}\s+\w+\s+\d{4})",
        ]
        for pat in date_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result.invoice_date_raw = m.group(1).strip()
                break

        # Due date
        due_match = re.search(
            r"(?:due|payment|pay\s*by)\s*(?:date)?\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            text, re.IGNORECASE
        )
        if due_match:
            result.due_date_raw = due_match.group(1).strip()

        # Seller / buyer blocks (rough extraction)
        bill_to_match = re.search(
            r"(?:bill\s*to|buyer|sold\s*to|ship\s*to)\s*[:\-]?\s*\n?(.*?)(?:\n\n|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if bill_to_match:
            result.buyer_block = bill_to_match.group(1).strip()[:200]

        seller_match = re.search(
            r"(?:from|seller|vendor|supplier)\s*[:\-]?\s*\n?(.*?)(?:\n\n|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if seller_match:
            result.seller_block = seller_match.group(1).strip()[:200]

        # Fallback: line-by-line search for invoice number (handles odd layouts)
        if not result.invoice_number_raw:
            for line in lines:
                m = re.search(r"(?:invoice|inv).{0,30}(?:no|number|#|num)\b.*[:\-]?\s*(\S+)", line, re.IGNORECASE)
                if m:
                    result.invoice_number_raw = m.group(1).strip()
                    break

    # ── Line Items Parsing ──────────────────────

    def _parse_line_items(self, result: StructuredInvoice, lines: list[str]) -> None:
        """
        Extract line items from table section.
        Handles tab-separated, pipe-separated, or space-aligned columns.
        """
        items = []
        # Skip the table header row
        data_lines = [l for l in lines if not self._is_table_header(l)]

        for line in data_lines:
            line = line.strip()
            if not line or self._is_separator(line):
                continue

            item = self._parse_single_line_item(line)
            if item:
                items.append(item)

        result.line_items = items

    def _parse_single_line_item(self, line: str) -> Optional[LineItem]:
        """Try to extract description, qty, rate, amount from a single line."""
        # Try pipe-separated
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            return self._build_line_item(parts)

        # Try tab-separated
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 3:
            return self._build_line_item(parts)

        # Try to split by multiple spaces (space-aligned columns)
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 2:
            return self._build_line_item(parts)

        return None

    def _build_line_item(self, parts: list[str]) -> Optional[LineItem]:
        """Build a LineItem from column parts, parsing numbers where possible."""
        if not parts:
            return None

        item = LineItem(description=parts[0])

        def parse_num(s: str) -> Optional[float]:
            cleaned = re.sub(r"[₹,\s]", "", s)
            try:
                return float(cleaned)
            except ValueError:
                return None

        if len(parts) >= 4:
            item.quantity = parse_num(parts[1])
            item.unit_price = parse_num(parts[2])
            item.total = parse_num(parts[3])
        elif len(parts) == 3:
            item.quantity = parse_num(parts[1])
            item.total = parse_num(parts[2])
        elif len(parts) == 2:
            item.total = parse_num(parts[1])

        # Must have a description
        if not item.description or len(item.description) < 2:
            return None
        return item

    # ── Totals Parsing ──────────────────────────

    def _parse_totals(self, result: StructuredInvoice, lines: list[str]) -> None:
        """Extract subtotal, GST/tax, and grand total from totals section."""
        text = "\n".join(lines)

        # Subtotal
        sub = re.search(
            r"(?:subtotal|sub\s*total|net\s*amount)\s*[:\-]?\s*([\d,\.]+)",
            text, re.IGNORECASE
        )
        if sub:
            result.subtotal_raw = sub.group(1).replace(",", "")

        # Tax (GST / CGST / SGST / IGST / VAT)
        tax = re.search(
            r"(?:gst|cgst|sgst|igst|tax|vat)\s*(?:@\s*[\d\.]+%)?\s*[:\-]?\s*([\d,\.]+)",
            text, re.IGNORECASE
        )
        if tax:
            result.tax_raw = tax.group(1).replace(",", "")

        # Total (grab the largest number near "total" keyword)
        total = re.search(
            r"(?:grand\s*total|total\s*amount|total\s*due|balance\s*due|total)\s*[:\-]?\s*([\d,\.]+)",
            text, re.IGNORECASE
        )
        if total:
            result.total_raw = total.group(1).replace(",", "")

    # ── Utilities ───────────────────────────────

    def _normalize_lines(self, text: str) -> list[str]:
        """Split text into lines, remove empty lines at start/end."""
        lines = text.splitlines()
        return [l.rstrip() for l in lines if l.strip()]

    def _is_table_header(self, line: str) -> bool:
        """Check if this line looks like a table header row."""
        ll = line.lower()
        return bool(re.search(r"desc|particular|qty|quantity|rate|amount|s\.?no", ll))

    def _is_separator(self, line: str) -> bool:
        """Check if line is a visual separator (---, ===, ...)."""
        stripped = line.replace(" ", "")
        return len(stripped) > 3 and all(c in "-=_*│─" for c in stripped)

    def _score_layout(self, result: StructuredInvoice) -> float:
        """Score how well we parsed the invoice (0–1)."""
        score = 0.0
        if result.invoice_number_raw: score += 0.2
        if result.invoice_date_raw:   score += 0.2
        if result.buyer_block:        score += 0.2
        if result.total_raw:          score += 0.2
        if result.line_items:         score += 0.2
        return score

    def to_dict(self, result: StructuredInvoice) -> dict:
        """Serialize StructuredInvoice to JSON-compatible dict."""
        return {
            "invoice_number_raw": result.invoice_number_raw,
            "invoice_date_raw": result.invoice_date_raw,
            "due_date_raw": result.due_date_raw,
            "seller_block": result.seller_block,
            "buyer_block": result.buyer_block,
            "line_items": [
                {
                    "description": li.description,
                    "quantity": li.quantity,
                    "unit_price": li.unit_price,
                    "total": li.total,
                }
                for li in result.line_items
            ],
            "subtotal_raw": result.subtotal_raw,
            "tax_raw": result.tax_raw,
            "total_raw": result.total_raw,
            "layout_confidence": result.layout_confidence,
            "section_boundaries": result.section_boundaries,
        }
