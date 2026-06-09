# Architecture Documentation

## System Overview

The Invoice AI system is a multi-stage pipeline that processes invoice documents
through six sequential transformations before producing a tagged, risk-classified output.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend (main.py)                  │
│  POST /upload  POST /demo/mock  GET /invoices  GET /stats        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │          OCR Pipeline              │
         │  • pdfplumber (text PDFs)          │
         │  • pdf2image + Tesseract (scanned) │
         │  • ImagePreprocessor (cv2)          │
         │  Output: OCRResult{text, confidence}│
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │         Text Structurer            │
         │  • Keyword-based section detection │
         │  • Table header detection          │
         │  • Proximity heuristics            │
         │  Output: StructuredInvoice{        │
         │    header, line_items, totals}     │
         └─────────────────┬─────────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │         Field Extractor            │
         │  • Regex patterns (invoice no,     │
         │    dates, amounts, GST)            │
         │  • spaCy NER (buyer/seller names)  │
         │  • Per-field confidence scores     │
         │  Output: InvoiceFields{            │
         │    invoice_number, buyer_name,     │
         │    total_amount, ...}              │
         └──────────────────┬────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
┌───────────▼───────────┐   ┌──────────────▼───────────┐
│   Anomaly Detector    │   │      Buyer Matcher        │
│                       │   │                           │
│ Checks:               │   │ • Exact match             │
│ • Duplicate invoice # │   │ • Alias table lookup      │
│ • Same buyer+amount   │   │ • Fuzzy match (RapidFuzz) │
│ • High/low amounts    │   │   - token_set_ratio        │
│   (Z-score threshold) │   │   - partial_ratio          │
│ • Date inconsistency  │   │   - WRatio                 │
│ • Missing fields      │   │ Output: BuyerMatch{        │
│ • Unknown buyer       │   │   buyer_id, match_score,  │
│                       │   │   match_method, profile}  │
│ Output: AnomalyReport │   └──────────────┬────────────┘
│ {anomalies, severity} │                  │
└───────────┬───────────┘                  │
            └───────────────┬──────────────┘
                            │
         ┌──────────────────▼────────────────┐
         │        Risk Classifier             │
         │  Rules (evaluated in order):       │
         │  1. High anomaly → HIGH            │
         │  2. Unknown buyer → HIGH           │
         │  3. Bad payment history → HIGH     │
         │  4. Medium anomaly → MEDIUM        │
         │  5. New buyer (<5 invoices) → MED  │
         │  6. Average payment history → MED  │
         │  7. None of above → LOW            │
         │  Output: RiskTag{                  │
         │    bucket, score, rationale}       │
         └──────────────────┬────────────────┘
                            │
         ┌──────────────────▼────────────────┐
         │         Structured JSON Output     │
         │  {                                 │
         │    invoice_number, buyer_name,     │
         │    total_amount, risk_bucket,      │
         │    anomalies[], rationale,         │
         │    confidence_scores{}             │
         │  }                                 │
         └───────────────────────────────────┘
```

## Data Flow Example

Input: `invoice.pdf`

```json
// After OCR
{
  "raw_text": "INVOICE\nInvoice No: BM-2024-09001\n...",
  "confidence": 0.92,
  "source_type": "pdf_text"
}

// After Text Structurer
{
  "invoice_number_raw": "BM-2024-09001",
  "invoice_date_raw": "01/09/2024",
  "buyer_block": "Reliance Retail Ltd\nNavi Mumbai...",
  "total_raw": "564630",
  "layout_confidence": 0.8
}

// After Field Extractor
{
  "invoice_number": {"value": "BM-2024-09001", "confidence": 0.9, "method": "regex"},
  "buyer_name": {"value": "Reliance Retail Ltd", "confidence": 0.85, "method": "regex_labeled"},
  "total_amount": {"value": "564630", "confidence": 0.9, "method": "regex_labeled"},
  "overall_confidence": 0.82
}

// After Anomaly Detector
{
  "is_clean": true,
  "highest_severity": "none",
  "risk_score": 0.0,
  "anomalies": []
}

// After Buyer Matcher + Risk Classifier
{
  "buyer_id": "B001",
  "canonical_name": "Reliance Retail Ltd",
  "match_method": "exact",
  "risk_bucket": "low",
  "rationale": "Known buyer with good payment history | No anomalies detected"
}
```

## Technology Choices

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| OCR | pytesseract + pdfplumber | Open-source, handles both text and scanned PDFs |
| Image preprocessing | OpenCV | Industry standard for image enhancement |
| NLP | spaCy en_core_web_sm | Fast, accurate NER for entity extraction |
| Fuzzy matching | RapidFuzz | 10x faster than fuzzywuzzy, better algorithms |
| API | FastAPI + uvicorn | Auto-docs, async support, type-checked |
| Date parsing | python-dateutil | Handles 50+ date formats automatically |
| Testing | pytest | Industry standard, clean fixture system |

## Anomaly Detection: Statistical Approach

For amount anomaly detection, the system uses Z-score thresholding:

```
Z = (x - μ) / σ

where:
  x = current invoice amount
  μ = buyer's historical average invoice amount
  σ = standard deviation of buyer's invoice amounts

Flags:
  Z > 2.0  → ABNORMALLY_HIGH_AMOUNT (high severity)
  Z < -2.0 → ABNORMALLY_LOW_AMOUNT  (medium severity)
```

This approach adapts to each buyer's individual invoice profile, so a
₹10L invoice for Flipkart is normal but unusual for Zomato.

## Buyer Matching: Multi-Stage Strategy

```
Input: "Reliance Retail"
         │
         ▼
    Normalize name
    "reliance retail"
         │
         ▼
    Exact match lookup ──── FOUND → return (score=1.0, method="exact")
         │
         │ NOT FOUND
         ▼
    Alias table lookup ──── FOUND → return (score=0.95, method="alias")
         │
         │ NOT FOUND
         ▼
    RapidFuzz (3 scorers)
    - token_set_ratio
    - partial_ratio
    - WRatio
         │
    score >= 75? ──── YES → return (score, method="fuzzy")
         │
         │ NO
         ▼
    return (buyer_id=None, method="none")
    → triggers UNKNOWN_BUYER anomaly
```
