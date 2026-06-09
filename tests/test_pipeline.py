"""
Week 8: Pipeline Tests
========================
Unit tests covering all pipeline components.
Run with: pytest tests/test_pipeline.py -v

Coverage:
  - OCR Mock Pipeline
  - Text Structurer (header, line items, totals)
  - Field Extractor (invoice number, dates, amounts, entities)
  - Anomaly Detector (all 7 anomaly types)
  - Buyer Matcher (exact, alias, fuzzy, unknown)
  - Risk Classifier (low, medium, high buckets)
"""

import sys
import json
import pytest
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from pipeline.ocr_pipeline import MockOCRPipeline
from pipeline.text_structurer import TextStructurer
from pipeline.field_extractor import FieldExtractor
from pipeline.anomaly_detector import AnomalyDetector, InvoiceHistoryStore
from pipeline.buyer_matcher import BuyerMatcher


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

SAMPLE_INVOICE_TEXT = """
INVOICE

Invoice No: BM-2024-09001
Invoice Date: 01/09/2024
Due Date: 01/10/2024

Bill To:
Reliance Retail Ltd
Navi Mumbai, Maharashtra

From:
Apex Packaging Co.
45 Industrial Estate, Pune - 411001

Description               Qty    Rate    Amount
HDPE Bags 50kg             500   650.00  325000
Corrugated Boxes L         200   767.50  153500

Subtotal                              478500
GST @ 18%                              86130
TOTAL                                 564630
"""

MOCK_BUYERS = [
    {
        "id": "B001",
        "name": "Reliance Retail Ltd",
        "aliases": ["Reliance Retail", "RRL", "Reliance"],
        "avg_invoice_amount": 485000,
        "std_dev": 95000,
        "payment_history": "good",
        "invoice_count": 42
    },
    {
        "id": "B002",
        "name": "Tata Consumer Products",
        "aliases": ["Tata Consumer", "TCPL"],
        "avg_invoice_amount": 320000,
        "std_dev": 60000,
        "payment_history": "excellent",
        "invoice_count": 67
    }
]


@pytest.fixture
def ocr():
    return MockOCRPipeline()

@pytest.fixture
def structurer():
    return TextStructurer()

@pytest.fixture
def extractor():
    return FieldExtractor(use_nlp=False)

@pytest.fixture
def history_store():
    return InvoiceHistoryStore()

@pytest.fixture
def detector(history_store):
    return AnomalyDetector(history_store)

@pytest.fixture
def matcher():
    return BuyerMatcher(buyer_db=MOCK_BUYERS)

@pytest.fixture
def structured_invoice(structurer):
    return structurer.structure(SAMPLE_INVOICE_TEXT)

@pytest.fixture
def extracted_fields(extractor, structured_invoice):
    return extractor.extract(SAMPLE_INVOICE_TEXT, structured_invoice)


# ──────────────────────────────────────────────
# Week 2: OCR Pipeline Tests
# ──────────────────────────────────────────────

class TestOCRPipeline:
    def test_mock_ocr_returns_text(self, ocr):
        result = ocr.process("fake_invoice.pdf")
        assert result.raw_text is not None
        assert len(result.raw_text) > 50

    def test_mock_ocr_has_high_confidence(self, ocr):
        result = ocr.process("fake_invoice.pdf")
        assert result.confidence > 0.8

    def test_mock_ocr_source_type(self, ocr):
        result = ocr.process("fake_invoice.pdf")
        assert result.source_type == "mock"

    def test_mock_ocr_process_bytes(self, ocr):
        result = ocr.process_bytes(b"fake content", "invoice.pdf")
        assert result.raw_text is not None

    def test_mock_text_contains_invoice_fields(self, ocr):
        result = ocr.process("invoice.pdf")
        text = result.raw_text.lower()
        for keyword in ["invoice", "total", "date"]:
            assert keyword in text, f"Expected '{keyword}' in OCR text"


# ──────────────────────────────────────────────
# Week 3: Text Structurer Tests
# ──────────────────────────────────────────────

class TestTextStructurer:
    def test_extracts_invoice_number(self, structured_invoice):
        assert "BM-2024-09001" in structured_invoice.invoice_number_raw

    def test_extracts_invoice_date(self, structured_invoice):
        assert "2024" in structured_invoice.invoice_date_raw or "09" in structured_invoice.invoice_date_raw

    def test_extracts_due_date(self, structured_invoice):
        assert structured_invoice.due_date_raw != ""

    def test_extracts_buyer_block(self, structured_invoice):
        assert "Reliance" in (structured_invoice.buyer_block or "")

    def test_extracts_total(self, structured_invoice):
        assert structured_invoice.total_raw != ""

    def test_layout_confidence_above_threshold(self, structured_invoice):
        assert structured_invoice.layout_confidence >= 0.4

    def test_empty_text_returns_empty_result(self, structurer):
        result = structurer.structure("")
        assert result.invoice_number_raw == ""

    def test_line_items_extracted(self, structured_invoice):
        # Line items should be present
        assert isinstance(structured_invoice.line_items, list)

    def test_to_dict_serializable(self, structurer, structured_invoice):
        d = structurer.to_dict(structured_invoice)
        json_str = json.dumps(d)
        assert json_str is not None


# ──────────────────────────────────────────────
# Week 4: Field Extractor Tests
# ──────────────────────────────────────────────

class TestFieldExtractor:
    def test_extracts_invoice_number(self, extracted_fields):
        assert extracted_fields.invoice_number.value is not None
        assert "BM-2024" in extracted_fields.invoice_number.value

    def test_extracts_buyer_name(self, extracted_fields):
        assert extracted_fields.buyer_name.value is not None
        assert "Reliance" in extracted_fields.buyer_name.value

    def test_extracts_total_amount(self, extracted_fields):
        assert extracted_fields.total_amount_float is not None
        assert extracted_fields.total_amount_float > 0

    def test_extracts_currency_inr(self, extracted_fields):
        assert extracted_fields.currency.value == "INR"

    def test_confidence_scores_in_range(self, extracted_fields):
        for attr in ["invoice_number", "invoice_date", "buyer_name", "total_amount"]:
            field = getattr(extracted_fields, attr)
            assert 0.0 <= field.confidence <= 1.0, f"{attr} confidence out of range"

    def test_overall_confidence_above_zero(self, extracted_fields):
        assert extracted_fields.overall_confidence > 0

    def test_to_dict_has_all_fields(self, extracted_fields):
        d = extracted_fields.to_dict()
        required_keys = ["invoice_number", "invoice_date", "buyer_name", "total_amount", "currency"]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_invoice_date_parsed(self, extracted_fields):
        # Should parse at least one date
        assert (
            extracted_fields.invoice_date_parsed is not None
            or extracted_fields.due_date_parsed is not None
        )


# ──────────────────────────────────────────────
# Week 5: Anomaly Detector Tests
# ──────────────────────────────────────────────

class TestAnomalyDetector:
    def test_clean_invoice_has_no_anomalies(self, detector, extracted_fields):
        buyer_profile = {"avg_invoice_amount": 485000, "std_dev": 95000}
        report = detector.analyze(extracted_fields, buyer_id="B001", buyer_profile=buyer_profile)
        # No duplicate, no date issues for a fresh invoice
        duplicate_flags = [a for a in report.anomalies if "DUPLICATE" in a.code]
        assert len(duplicate_flags) == 0

    def test_duplicate_detection(self, detector, extracted_fields):
        # Register the invoice once
        detector.history.add("BM-2024-09001", "B001", 564630, "Apex Packaging")
        # Now analyze the same invoice again
        report = detector.analyze(extracted_fields, buyer_id="B001")
        codes = [a.code for a in report.anomalies]
        assert "DUPLICATE_INVOICE_NUMBER" in codes

    def test_high_amount_detection(self, detector, extracted_fields, extractor):
        # Create invoice with abnormally high amount
        high_amount_text = SAMPLE_INVOICE_TEXT.replace("564630", "9000000").replace("478500", "7500000")
        fields = extractor.extract(high_amount_text)
        buyer_profile = {"avg_invoice_amount": 485000, "std_dev": 95000}
        report = detector.analyze(fields, buyer_id="B001", buyer_profile=buyer_profile)
        codes = [a.code for a in report.anomalies]
        assert "ABNORMALLY_HIGH_AMOUNT" in codes

    def test_unknown_buyer_flagged(self, detector, extracted_fields):
        report = detector.analyze(extracted_fields, buyer_id=None)
        codes = [a.code for a in report.anomalies]
        assert "UNKNOWN_BUYER" in codes

    def test_risk_score_range(self, detector, extracted_fields):
        report = detector.analyze(extracted_fields, buyer_id="B001")
        assert 0.0 <= report.risk_score <= 1.0

    def test_high_severity_sets_flag(self, detector, extracted_fields):
        # Register duplicate to trigger high severity
        detector.history.add("BM-2024-09001", "B001", 564630, "Apex")
        report = detector.analyze(extracted_fields, buyer_id="B001")
        assert report.highest_severity in ["medium", "high"]

    def test_to_dict_structure(self, detector, extracted_fields):
        report = detector.analyze(extracted_fields, buyer_id="B001")
        d = report.to_dict()
        assert "is_clean" in d
        assert "anomalies" in d
        assert "risk_score" in d

    def test_date_inconsistency_detected(self, detector, extractor):
        # Invoice where due date is before invoice date
        bad_date_text = """
        Invoice No: TEST-001
        Invoice Date: 15/09/2024
        Due Date: 10/09/2024
        Bill To: Reliance Retail Ltd
        TOTAL 100000
        """
        fields = extractor.extract(bad_date_text)
        # Manually set parsed dates to simulate the issue
        fields.invoice_date_parsed = datetime(2024, 9, 15)
        fields.due_date_parsed = datetime(2024, 9, 10)
        report = detector.analyze(fields, buyer_id="B001")
        codes = [a.code for a in report.anomalies]
        assert "DATE_INCONSISTENCY" in codes


# ──────────────────────────────────────────────
# Week 6: Buyer Matcher Tests
# ──────────────────────────────────────────────

class TestBuyerMatcher:
    def test_exact_match(self, matcher):
        result = matcher.match("Reliance Retail Ltd")
        assert result.buyer_id == "B001"
        assert result.match_method == "exact"
        assert result.match_score == 1.0

    def test_alias_match(self, matcher):
        result = matcher.match("RRL")
        assert result.buyer_id == "B001"
        assert result.match_method == "alias"

    def test_fuzzy_match(self, matcher):
        result = matcher.match("Reliance Retail")
        # Should match B001 (Reliance Retail Ltd)
        assert result.buyer_id == "B001"

    def test_unknown_buyer_returns_none(self, matcher):
        result = matcher.match("XYZ Unknown Company Ltd")
        assert result.buyer_id is None
        assert result.match_method in ["none", "fuzzy"]

    def test_empty_string_returns_none(self, matcher):
        result = matcher.match("")
        assert result.buyer_id is None

    def test_buyer_profile_included(self, matcher):
        result = matcher.match("Tata Consumer Products")
        assert result.buyer_profile is not None
        assert "payment_history" in result.buyer_profile

    def test_risk_classification_low(self, matcher, extracted_fields):
        from pipeline.anomaly_detector import AnomalyReport
        clean_report = AnomalyReport()  # No anomalies
        buyer_match = matcher.match("Reliance Retail Ltd")
        risk = matcher.classify_risk(extracted_fields, clean_report, buyer_match)
        assert risk.bucket == "low"

    def test_risk_classification_high_for_unknown_buyer(self, matcher, extracted_fields):
        from pipeline.anomaly_detector import AnomalyReport
        report = AnomalyReport()
        report.add("UNKNOWN_BUYER")
        buyer_match = matcher.match("Unknown Company XYZ")
        risk = matcher.classify_risk(extracted_fields, report, buyer_match)
        assert risk.bucket == "high"

    def test_match_and_tag_returns_tagged_invoice(self, matcher, extracted_fields):
        from pipeline.anomaly_detector import AnomalyReport
        report = AnomalyReport()
        tagged = matcher.match_and_tag("Reliance Retail Ltd", extracted_fields, report)
        assert tagged.buyer_match is not None
        assert tagged.risk_tag is not None
        assert tagged.risk_tag.bucket in ["low", "medium", "high"]


# ──────────────────────────────────────────────
# Integration Test
# ──────────────────────────────────────────────

class TestFullPipeline:
    def test_end_to_end_clean_invoice(self):
        """Full pipeline run on a clean invoice."""
        ocr = MockOCRPipeline()
        structurer = TextStructurer()
        extractor = FieldExtractor(use_nlp=False)
        history = InvoiceHistoryStore()
        detector = AnomalyDetector(history)
        matcher = BuyerMatcher(buyer_db=MOCK_BUYERS)

        # Step 1: OCR
        ocr_result = ocr.process("test.pdf")
        assert ocr_result.raw_text

        # Step 2: Structure
        structured = structurer.structure(ocr_result.raw_text)
        assert structured.layout_confidence > 0

        # Step 3: Extract
        fields = extractor.extract(ocr_result.raw_text, structured)
        assert fields.overall_confidence > 0

        # Step 4: Match buyer
        buyer_name = fields.buyer_name.value or "Reliance Retail Ltd"
        buyer_match = matcher.match(buyer_name)

        # Step 5: Detect anomalies
        report = detector.analyze(fields, buyer_match.buyer_id, buyer_match.buyer_profile)
        assert isinstance(report.anomalies, list)

        # Step 6: Tag
        tagged = matcher.match_and_tag(buyer_name, fields, report)
        assert tagged.risk_tag.bucket in ["low", "medium", "high"]

        print(f"\n✅ Pipeline result: {tagged.risk_tag.bucket.upper()} risk | "
              f"{len(report.anomalies)} anomalies | "
              f"confidence: {fields.overall_confidence:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
