# 🧾 Invoice Understanding & Auto-Tagging AI
>
> **BillMart Internship Project** — 8-Week Build Plan

An end-to-end AI system that reads invoices in multiple formats, extracts key fields, detects anomalies, and auto-tags invoices by buyer and risk bucket — with explainable outputs.

---

## 📁 Project Structure

```text
invoice-ai/
├── backend/
│   ├── main.py                  # FastAPI app — main entry point
│   ├── pipeline/
│   │   ├── ocr_pipeline.py      # Week 2: OCR for PDF & images
│   │   ├── text_structurer.py   # Week 3: Layout parsing → JSON blocks
│   │   ├── field_extractor.py   # Week 4: NLP entity extraction
│   │   ├── anomaly_detector.py  # Week 5: Duplicate & risk detection
│   │   └── buyer_matcher.py     # Week 6: Fuzzy match + risk bucketing
│   └── utils/
│       ├── mock_data.py         # Realistic invoice mock dataset
│       └── helpers.py           # Shared utilities
├── frontend/
│   ├── index.html               # Dashboard UI
│   ├── css/style.css            # Styles
│   └── js/app.js                # Frontend logic
├── data/
│   └── sample_invoices.json     # Mock invoice dataset
├── tests/
│   └── test_pipeline.py         # Unit tests
├── docs/
│   └── architecture.md          # System design doc
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 3. Open the Frontend

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
```

Then visit: `http://localhost:3000`

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
| -------- | ---------- | ------------- |
| `POST` | `/upload` | Upload invoice (PDF/image) |
| `GET` | `/invoices` | List all processed invoices |
| `GET` | `/invoices/{id}` | Get single invoice details |
| `GET` | `/anomalies` | List flagged anomalies |
| `GET` | `/buyers` | List known buyers + risk |
| `POST` | `/demo/mock` | Generate mock invoice for demo |

---

## 🧪 Weekly Deliverables Checklist

| Week | Focus | Status |
| ------ | ------- | -------- |
| 1 | Problem understanding & dataset setup | ✅ |
| 2 | OCR pipeline implementation | ✅ |
| 3 | Text structuring & layout understanding | ✅ |
| 4 | NLP-based field extraction | ✅ |
| 5 | Anomaly detection engine | ✅ |
| 6 | Buyer matching & auto-tagging | ✅ |
| 7 | End-to-end integration & UI | ✅ |
| 8 | Testing, documentation & final demo | ✅ |

---

## 🏗️ Architecture Overview

```text
Invoice (PDF/Image)
        │
        ▼
   OCR Pipeline
  (pytesseract / pdfplumber)
        │
        ▼
  Text Structurer
  (header / line items / totals)
        │
        ▼
  Field Extractor
  (regex + spaCy NLP)
        │
        ├──────────────────────┐
        ▼                      ▼
 Anomaly Detector        Buyer Matcher
 (duplicates, amounts)   (fuzzy match)
        │                      │
        └──────────┬───────────┘
                   ▼
         Risk Classification
         (Low / Medium / High)
                   │
                   ▼
         Structured JSON Output
         + Dashboard UI
```

---

## 📊 Key Invoice Fields Extracted

- **Invoice Number** — unique identifier
- **Invoice Date** — date of issue
- **Due Date** — payment deadline
- **Buyer Name** — purchasing entity
- **Seller Name** — supplying entity
- **Invoice Amount** — total payable
- **GST / Tax** — applicable tax (optional)
- **Line Items** — goods/services listed

## 🚨 Anomaly Types Detected

1. **Duplicate Invoice Number** — same invoice ID submitted twice
2. **Same Invoice Re-uploaded** — identical content, different file
3. **Abnormally High Amount** — >3σ above buyer's historical average
4. **Abnormally Low Amount** — suspiciously low for buyer profile
5. **Missing Required Fields** — incomplete invoice
6. **Date Inconsistencies** — due date before invoice date

## 🏷️ Risk Buckets

| Bucket | Criteria |
| ------ | ---------- |
| 🟢 Low | Known buyer, normal amount, clean history |
| 🟡 Medium | New buyer OR amount slightly above average |
| 🔴 High | Unknown buyer, anomaly flagged, large amount |
