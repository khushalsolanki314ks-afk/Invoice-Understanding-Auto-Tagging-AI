"""
Week 7: End-to-End FastAPI Application
=========================================
Orchestrates the full invoice processing pipeline:
  Upload → OCR → Structure → Extract → Anomaly Detection → Buyer Match → Tag

Endpoints:
  POST /upload         — Process a real invoice file
  POST /demo/mock      — Generate a demo result from mock data
  GET  /invoices       — List all processed invoices
  GET  /invoices/{id}  — Get single invoice details
  GET  /anomalies      — List flagged invoices
  GET  /buyers         — List buyers with risk stats
  GET  /stats          — Dashboard summary statistics

Run with:
  uvicorn main:app --reload --port 8000
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

# ── Pipeline imports ──────────────────────────
from pipeline.ocr_pipeline import MockOCRPipeline, OCRPipeline
from pipeline.text_structurer import TextStructurer
from pipeline.field_extractor import FieldExtractor
from pipeline.anomaly_detector import AnomalyDetector, InvoiceHistoryStore
from pipeline.buyer_matcher import BuyerMatcher


# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Invoice AI — BillMart",
    description="Invoice Understanding & Auto-Tagging AI System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (replace with DB in production) ─
DATA_FILE = Path(__file__).parent.parent / "data" / "sample_invoices.json"
processed_invoices: list[dict] = []
history_store = InvoiceHistoryStore()

# ── Initialize pipeline components ──
structurer = TextStructurer()
extractor = FieldExtractor(use_nlp=False)      # Set True when spaCy is installed
anomaly_detector = AnomalyDetector(history_store)
buyer_matcher = BuyerMatcher()

# Pre-load mock invoices at startup
def _load_mock_data() -> None:
    global processed_invoices
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
        processed_invoices = data.get("invoices", [])
        # Register history for anomaly detection
        for inv in processed_invoices:
            if inv.get("invoice_number") and inv.get("invoice_number") not in ["BM-2024-09001"]:
                history_store.add(
                    inv["invoice_number"],
                    inv.get("buyer_id"),
                    inv.get("amount"),
                    inv.get("seller_name"),
                )
        logger.info(f"Loaded {len(processed_invoices)} mock invoices")
    except Exception as e:
        logger.error(f"Failed to load mock data: {e}")
        processed_invoices = []

_load_mock_data()


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────

class ProcessingResult(BaseModel):
    invoice_id: str
    status: str
    invoice_number: Optional[str]
    invoice_date: Optional[str]
    due_date: Optional[str]
    buyer_name: Optional[str]
    seller_name: Optional[str]
    amount: Optional[float]
    gst_amount: Optional[float]
    total_amount: Optional[float]
    currency: str
    risk_bucket: str
    risk_score: float
    anomalies: list[dict]
    buyer_match: dict
    confidence: float
    processed_at: str


# ──────────────────────────────────────────────
# Core Processing Pipeline
# ──────────────────────────────────────────────

def run_pipeline(file_bytes: bytes, filename: str, lang: str = "eng", use_real_ocr: bool = True) -> dict:
    """
    Full invoice processing pipeline.
    Returns a dict with all extracted fields, anomalies, and tags.
    """
    invoice_id = str(uuid.uuid4())[:8].upper()
    logger.info(f"[{invoice_id}] Starting pipeline for: {filename}")

    # ── Step 1: Save upload & OCR ──────────────────────────
    import tempfile
    suffix = Path(filename).suffix or ''
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    logger.info(f"[{invoice_id}] Received upload: {filename} | size={len(file_bytes)} bytes | saved={tmp_path} | ts={datetime.now().isoformat()}")

    # Choose OCR engine
    if use_real_ocr:
        ocr_instance = OCRPipeline(lang=lang, preprocess=True)
    else:
        ocr_instance = MockOCRPipeline()

    try:
        # Prefer calling process(path) so OCR pipeline can log and operate on file path
        ocr_result = ocr_instance.process(tmp_path)
    except Exception as e:
        # Ensure temp file is removed
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        logger.error(f"[{invoice_id}] OCR processing failed: {e}", exc_info=True)
        raise

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        logger.debug(f"Could not delete tmp file: {tmp_path}")

    raw_text = ocr_result.raw_text
    logger.info(f"[{invoice_id}] OCR complete — source={ocr_result.source_type} | confidence={ocr_result.confidence:.2f}")
    logger.debug(f"[{invoice_id}] OCR raw text:\n{raw_text}\n---END OCR---")

    # ── Step 2: Text Structuring ──────────────
    structured = structurer.structure(raw_text)
    logger.info(f"[{invoice_id}] Structure complete — layout confidence: {structured.layout_confidence:.2f}")

    # ── Step 3: Field Extraction ──────────────
    fields = extractor.extract(raw_text, structured)
    try:
        logger.info(f"[{invoice_id}] Fields extracted — overall confidence: {fields.overall_confidence:.2f}")
        logger.debug(f"[{invoice_id}] Extracted fields: {fields.to_dict()}")
    except Exception:
        logger.exception("Failed to log extracted fields")

    # ── Step 4: Buyer Matching ────────────────
    raw_buyer = fields.buyer_name.value or ""
    buyer_match_result = buyer_matcher.match(raw_buyer)
    buyer_id = buyer_match_result.buyer_id
    buyer_profile = buyer_match_result.buyer_profile

    # ── Step 5: Anomaly Detection ─────────────
    anomaly_report = anomaly_detector.analyze(fields, buyer_id, buyer_profile)
    logger.info(f"[{invoice_id}] Anomalies: {len(anomaly_report.anomalies)} flags — {anomaly_report.highest_severity}")
    logger.debug(f"[{invoice_id}] Anomaly details: {anomaly_report.to_dict()}")

    # ── Step 6: Risk Tagging ──────────────────
    tagged = buyer_matcher.match_and_tag(raw_buyer, fields, anomaly_report)
    risk = tagged.risk_tag
    logger.info(f"[{invoice_id}] Risk: {risk.bucket.upper()} (score: {risk.score:.2f})")

    # ── Register in history ───────────────────
    history_store.add(
        fields.invoice_number.value or invoice_id,
        buyer_id,
        fields.total_amount_float or fields.amount_float,
        fields.seller_name.value,
    )

    # ── Compose result ────────────────────────
    # If amount anomalies (high/low) exist, mark as rejected
    rejection_reasons = [a.code for a in anomaly_report.anomalies if a.code in ("ABNORMALLY_HIGH_AMOUNT", "ABNORMALLY_LOW_AMOUNT")]

    status = "flagged" if not anomaly_report.is_clean else "processed"
    if rejection_reasons:
        status = "rejected"

    result = {
        "id": f"INV-NEW-{invoice_id}",
        "invoice_id": invoice_id,
        "invoice_number": fields.invoice_number.value,
        "invoice_date": fields.invoice_date.value,
        "due_date": fields.due_date.value,
        "buyer_name": buyer_match_result.buyer_name or fields.buyer_name.value,
        "buyer_id": buyer_id,
        "seller_name": fields.seller_name.value,
        "amount": fields.amount_float,
        "gst": fields.gst_amount.value,
        "total": fields.total_amount_float,
        "currency": fields.currency.value,
        "risk_bucket": risk.bucket,
        "risk_score": risk.score,
        "risk_rationale": risk.rationale,
        "anomalies": [a["code"] for a in anomaly_report.to_dict()["anomalies"]],
        "anomaly_details": anomaly_report.to_dict(),
        "buyer_match": tagged.buyer_match.__dict__ if hasattr(tagged.buyer_match, "__dict__") else {},
        "extraction_confidence": fields.overall_confidence,
        "ocr_confidence": ocr_result.confidence,
        "ocr_source": ocr_result.source_type,
        "status": status,
        "rejection_reasons": rejection_reasons,
        "processed_at": datetime.now().isoformat(),
        "filename": filename,
        "line_items": [
            {
                "description": li.description,
                "quantity": li.quantity,
                "unit_price": li.unit_price,
                "total": li.total,
            }
            for li in structured.line_items
        ],
    }

    processed_invoices.append(result)
    logger.info(f"[{invoice_id}] Final response: invoice_number={result['invoice_number']} total={result['total']} status={result['status']}")
    return result


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Invoice AI — BillMart",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/upload", "/demo/mock", "/invoices", "/anomalies", "/buyers", "/stats"],
    }


@app.post("/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    lang: str = Query(default="eng", description="Tesseract language code, e.g. eng, hin, fra"),
    use_real_ocr: bool = Query(default=True, description="Use real Tesseract OCR instead of mock"),
):
    """
    Upload an invoice file (PDF or image) for processing.
    Returns structured data, anomaly flags, and risk classification.
    """
    allowed_types = {"application/pdf", "image/png", "image/jpeg", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use PDF, PNG, or JPEG."
        )

    if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    contents = await file.read()
    try:
        result = run_pipeline(contents, file.filename, lang=lang, use_real_ocr=use_real_ocr)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/demo/mock")
def demo_mock(buyer_scenario: str = Query(default="clean", description="Scenario: clean|duplicate|high_amount|unknown_buyer")):
    """
    Generate a demo invoice processing result without uploading a file.
    Useful for UI testing and demonstration.
    """
    # Reload mock data to simulate the pipeline
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)

        scenario_map = {
            "clean": 0,
            "duplicate": 2,
            "high_amount": 3,
            "date_issue": 5,
            "unknown_buyer": 6,
        }
        idx = scenario_map.get(buyer_scenario, 0)
        mock_inv = data["invoices"][idx]

        # Enrich with pipeline-like structure
        mock_inv["ocr_confidence"] = 0.92
        mock_inv["extraction_confidence"] = 0.88
        mock_inv["ocr_source"] = "mock"
        mock_inv["processed_at"] = datetime.now().isoformat()
        mock_inv["anomaly_details"] = {
            "is_clean": len(mock_inv.get("anomalies", [])) == 0,
            "highest_severity": "high" if mock_inv.get("risk_bucket") == "high" else "medium" if mock_inv.get("risk_bucket") == "medium" else "none",
            "risk_score": {"low": 0.1, "medium": 0.5, "high": 0.85}.get(mock_inv.get("risk_bucket", "low"), 0.1),
            "anomaly_count": len(mock_inv.get("anomalies", [])),
            "anomalies": [
                {
                    "code": code,
                    "severity": "high" if code in ["DUPLICATE_INVOICE_NUMBER", "ABNORMALLY_HIGH_AMOUNT", "UNKNOWN_BUYER"] else "medium",
                    "description": {
                        "DUPLICATE_INVOICE_NUMBER": "Invoice number submitted before",
                        "SAME_AMOUNT_SAME_BUYER": "Same amount for this buyer recently",
                        "ABNORMALLY_HIGH_AMOUNT": "Amount far above buyer average",
                        "DATE_INCONSISTENCY": "Due date is before invoice date",
                        "UNKNOWN_BUYER": "Buyer not in database",
                    }.get(code, code),
                    "action": "Manual review required",
                    "detail": "Detected during automated processing",
                }
                for code in mock_inv.get("anomalies", [])
            ],
        }
        return JSONResponse(content=mock_inv)
    except Exception as e:
        logger.error(f"Demo mock error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/invoices")
def list_invoices(
    status: Optional[str] = Query(None, description="Filter by status: processed|flagged"),
    risk: Optional[str] = Query(None, description="Filter by risk: low|medium|high"),
    limit: int = Query(50, ge=1, le=200),
):
    """List all processed invoices with optional filters."""
    result = processed_invoices
    if status:
        result = [i for i in result if i.get("status") == status]
    if risk:
        result = [i for i in result if i.get("risk_bucket") == risk]
    return {"total": len(result), "invoices": result[:limit]}


@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    """Get full details of a single invoice."""
    for inv in processed_invoices:
        if inv.get("id") == invoice_id or inv.get("invoice_id") == invoice_id:
            return inv
    raise HTTPException(status_code=404, detail=f"Invoice not found: {invoice_id}")


@app.get("/anomalies")
def list_anomalies():
    """List all invoices with anomaly flags."""
    flagged = [i for i in processed_invoices if i.get("anomalies") or i.get("status") == "flagged"]
    return {
        "total": len(flagged),
        "flagged_invoices": flagged,
        "summary": {
            "high": len([i for i in flagged if i.get("risk_bucket") == "high"]),
            "medium": len([i for i in flagged if i.get("risk_bucket") == "medium"]),
        }
    }


@app.get("/buyers")
def list_buyers():
    """List all buyers with their risk profiles and invoice counts."""
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
        buyers = data.get("buyers", [])

        # Enrich with invoice stats
        for buyer in buyers:
            buyer_invoices = [i for i in processed_invoices if i.get("buyer_id") == buyer["id"]]
            buyer["invoice_count_current"] = len(buyer_invoices)
            buyer["flagged_count"] = len([i for i in buyer_invoices if i.get("status") == "flagged"])
            payment_risk_map = {"excellent": "low", "good": "low", "average": "medium", "poor": "high"}
            buyer["risk_bucket"] = payment_risk_map.get(buyer.get("payment_history", "average"), "medium")

        return {"total": len(buyers), "buyers": buyers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def dashboard_stats():
    """Summary statistics for the dashboard."""
    total = len(processed_invoices)
    flagged = len([i for i in processed_invoices if i.get("anomalies") or i.get("status") == "flagged"])
    clean = total - flagged

    by_risk = {
        "low": len([i for i in processed_invoices if i.get("risk_bucket") == "low"]),
        "medium": len([i for i in processed_invoices if i.get("risk_bucket") == "medium"]),
        "high": len([i for i in processed_invoices if i.get("risk_bucket") == "high"]),
    }

    total_value = sum(
        (i.get("total") or i.get("amount") or 0)
        for i in processed_invoices
    )

    return {
        "total_invoices": total,
        "clean_invoices": clean,
        "flagged_invoices": flagged,
        "by_risk_bucket": by_risk,
        "total_invoice_value": total_value,
        "average_invoice_value": total_value / total if total > 0 else 0,
        "flag_rate": flagged / total if total > 0 else 0,
    }
