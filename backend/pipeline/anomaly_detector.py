"""
Week 5: Anomaly Detection Engine
===================================
Detects risky or incorrect invoices by checking for:
  1. Duplicate invoice numbers
  2. Same invoice re-uploaded (same buyer + amount + seller)
  3. Abnormally high/low amounts (statistical thresholds)
  4. Missing required fields
  5. Date inconsistencies (due date before invoice date)
  6. Unknown buyer

Each anomaly comes with a reason code and human-readable explanation.

Usage:
    from pipeline.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(history_store)
    result = detector.analyze(invoice_fields, buyer_id="B001")
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from loguru import logger


# ──────────────────────────────────────────────
# Anomaly Types
# ──────────────────────────────────────────────

ANOMALY_CODES = {
    "DUPLICATE_INVOICE_NUMBER": {
        "severity": "high",
        "description": "This invoice number has already been submitted.",
        "action": "Reject or request seller clarification.",
    },
    "SAME_AMOUNT_SAME_BUYER": {
        "severity": "high",
        "description": "Identical amount for the same buyer was submitted recently.",
        "action": "Check for duplicate submission.",
    },
    "ABNORMALLY_HIGH_AMOUNT": {
        "severity": "high",
        "description": "Invoice amount is significantly above this buyer's historical average.",
        "action": "Manual review required before disbursement.",
    },
    "ABNORMALLY_LOW_AMOUNT": {
        "severity": "medium",
        "description": "Invoice amount is unusually low for this buyer.",
        "action": "Verify line items are complete.",
    },
    "MISSING_INVOICE_NUMBER": {
        "severity": "high",
        "description": "Invoice number could not be extracted.",
        "action": "Request original document from seller.",
    },
    "MISSING_BUYER_NAME": {
        "severity": "high",
        "description": "Buyer name not found in invoice.",
        "action": "Cannot process — buyer unknown.",
    },
    "MISSING_AMOUNT": {
        "severity": "high",
        "description": "Invoice amount could not be determined.",
        "action": "Request corrected invoice.",
    },
    "DATE_INCONSISTENCY": {
        "severity": "medium",
        "description": "Due date is on or before the invoice date.",
        "action": "Verify payment terms with seller.",
    },
    "PAST_DUE_DATE": {
        "severity": "medium",
        "description": "Invoice due date has already passed.",
        "action": "Check if payment was already made.",
    },
    "UNKNOWN_BUYER": {
        "severity": "medium",
        "description": "Buyer is not in the registered buyer database.",
        "action": "Onboard buyer before processing.",
    },
    "LOW_EXTRACTION_CONFIDENCE": {
        "severity": "low",
        "description": "Field extraction confidence is below threshold.",
        "action": "Manual review of extracted fields recommended.",
    },
}


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class Anomaly:
    code: str
    severity: str           # "low" | "medium" | "high"
    description: str
    action: str
    detail: str = ""        # Specific detail for this invoice

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "description": self.description,
            "action": self.action,
            "detail": self.detail,
        }


@dataclass
class AnomalyReport:
    anomalies: list[Anomaly] = field(default_factory=list)
    is_clean: bool = True
    highest_severity: str = "none"   # "none" | "low" | "medium" | "high"
    risk_score: float = 0.0          # 0.0–1.0 composite risk

    def add(self, code: str, detail: str = "") -> None:
        meta = ANOMALY_CODES.get(code, {"severity": "low", "description": code, "action": ""})
        self.anomalies.append(Anomaly(
            code=code,
            severity=meta["severity"],
            description=meta["description"],
            action=meta["action"],
            detail=detail,
        ))
        self.is_clean = False
        self._update_severity(meta["severity"])

    def _update_severity(self, new: str) -> None:
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        if order.get(new, 0) > order.get(self.highest_severity, 0):
            self.highest_severity = new

    def to_dict(self) -> dict:
        return {
            "is_clean": self.is_clean,
            "highest_severity": self.highest_severity,
            "risk_score": round(self.risk_score, 2),
            "anomaly_count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


# ──────────────────────────────────────────────
# Invoice History Store (In-memory for demo)
# ──────────────────────────────────────────────

class InvoiceHistoryStore:
    """
    Stores processed invoices for duplicate detection.
    In production, replace with a database query layer.
    """

    def __init__(self):
        # invoice_number → list of invoice summaries
        self._by_number: dict[str, list[dict]] = {}
        # buyer_id → list of amounts (for statistical baseline)
        self._amounts_by_buyer: dict[str, list[float]] = {}

    def add(self, invoice_number: str, buyer_id: Optional[str],
            amount: Optional[float], seller_name: Optional[str]) -> None:
        key = (invoice_number or "").upper().strip()
        if key:
            self._by_number.setdefault(key, []).append({
                "buyer_id": buyer_id,
                "amount": amount,
                "seller": seller_name,
            })
        if buyer_id and amount:
            self._amounts_by_buyer.setdefault(buyer_id, []).append(amount)

    def get_history_for_number(self, invoice_number: str) -> list[dict]:
        return self._by_number.get(invoice_number.upper().strip(), [])

    def get_buyer_stats(self, buyer_id: str) -> dict:
        amounts = self._amounts_by_buyer.get(buyer_id, [])
        if len(amounts) < 3:
            return {}
        mean = sum(amounts) / len(amounts)
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = math.sqrt(variance)
        return {"mean": mean, "std": std, "count": len(amounts), "amounts": amounts}


# ──────────────────────────────────────────────
# Anomaly Detector
# ──────────────────────────────────────────────

class AnomalyDetector:
    """
    Runs multiple anomaly checks on an extracted invoice.
    Returns an AnomalyReport with detailed flags and risk score.
    """

    # Amount threshold: flag if Z-score > this value
    HIGH_AMOUNT_ZSCORE = 2.0
    LOW_AMOUNT_ZSCORE = 2.0

    # Confidence threshold below which we flag for review
    MIN_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, history_store: Optional[InvoiceHistoryStore] = None):
        self.history = history_store or InvoiceHistoryStore()

    def analyze(
        self,
        invoice_fields,         # InvoiceFields from field_extractor
        buyer_id: Optional[str] = None,
        buyer_profile: Optional[dict] = None,
    ) -> AnomalyReport:
        """
        Run all anomaly checks and return an AnomalyReport.

        Args:
            invoice_fields: Extracted invoice fields
            buyer_id: Resolved buyer ID (from buyer_matcher)
            buyer_profile: Buyer dict with avg_invoice_amount, std_dev etc.
        """
        report = AnomalyReport()
        logger.info(f"Running anomaly detection for invoice: {invoice_fields.invoice_number.value}")

        # ── 1. Missing required fields ──
        self._check_missing_fields(report, invoice_fields)

        # ── 2. Duplicate invoice number ──
        self._check_duplicate_number(report, invoice_fields, buyer_id)

        # ── 3. Amount anomalies ──
        self._check_amount_anomaly(report, invoice_fields, buyer_id, buyer_profile)

        # ── 4. Date inconsistencies ──
        self._check_date_issues(report, invoice_fields)

        # ── 5. Unknown buyer ──
        self._check_unknown_buyer(report, buyer_id)

        # ── 6. Low extraction confidence ──
        self._check_confidence(report, invoice_fields)

        # ── Compute composite risk score ──
        report.risk_score = self._compute_risk_score(report)

        logger.info(
            f"Anomaly check complete — {len(report.anomalies)} flags | "
            f"severity: {report.highest_severity} | risk: {report.risk_score:.2f}"
        )
        return report

    # ── Individual Checks ───────────────────────

    def _check_missing_fields(self, report: AnomalyReport, fields) -> None:
        """Flag missing critical fields."""
        if not fields.invoice_number.value:
            report.add("MISSING_INVOICE_NUMBER")

        if not fields.buyer_name.value:
            report.add("MISSING_BUYER_NAME")

        if not fields.total_amount_float and not fields.amount_float:
            report.add("MISSING_AMOUNT")

    def _check_duplicate_number(
        self, report: AnomalyReport, fields, buyer_id: Optional[str]
    ) -> None:
        """Check for duplicate invoice numbers in history."""
        inv_no = fields.invoice_number.value
        if not inv_no:
            return

        history = self.history.get_history_for_number(inv_no)
        if history:
            report.add(
                "DUPLICATE_INVOICE_NUMBER",
                detail=f"Invoice #{inv_no} was previously submitted {len(history)} time(s)."
            )

            # Also check same buyer + amount
            amount = fields.total_amount_float or fields.amount_float
            for past in history:
                if past.get("buyer_id") == buyer_id and past.get("amount") == amount:
                    report.add(
                        "SAME_AMOUNT_SAME_BUYER",
                        detail=f"Exact match found: buyer {buyer_id}, amount ₹{amount:,.2f}"
                    )
                    break

    def _check_amount_anomaly(
        self, report: AnomalyReport, fields,
        buyer_id: Optional[str], buyer_profile: Optional[dict]
    ) -> None:
        """
        Detect abnormally high or low amounts using:
        1. Buyer profile stats (if available from buyer_matcher)
        2. Historical data in InvoiceHistoryStore
        """
        amount = fields.total_amount_float or fields.amount_float
        if not amount or amount <= 0:
            return

        # Get stats from buyer profile (buyer_matcher output)
        stats = None
        if buyer_profile:
            mean = buyer_profile.get("avg_invoice_amount")
            std = buyer_profile.get("std_dev")
            if mean and std:
                stats = {"mean": mean, "std": std}

        # Fall back to history store stats
        if not stats and buyer_id:
            stats = self.history.get_buyer_stats(buyer_id)

        if stats and stats.get("std", 0) > 0:
            z_score = (amount - stats["mean"]) / stats["std"]
            if z_score > self.HIGH_AMOUNT_ZSCORE:
                report.add(
                    "ABNORMALLY_HIGH_AMOUNT",
                    detail=(
                        f"Amount ₹{amount:,.0f} is {z_score:.1f}σ above "
                        f"buyer average ₹{stats['mean']:,.0f} (±₹{stats['std']:,.0f})"
                    )
                )
            elif z_score < -self.LOW_AMOUNT_ZSCORE:
                report.add(
                    "ABNORMALLY_LOW_AMOUNT",
                    detail=(
                        f"Amount ₹{amount:,.0f} is {abs(z_score):.1f}σ below "
                        f"buyer average ₹{stats['mean']:,.0f}"
                    )
                )

    def _check_date_issues(self, report: AnomalyReport, fields) -> None:
        """Check for date inconsistencies."""
        inv_date = fields.invoice_date_parsed
        due_date = fields.due_date_parsed
        now = datetime.now()

        # Due date before invoice date
        if inv_date and due_date and due_date <= inv_date:
            report.add(
                "DATE_INCONSISTENCY",
                detail=f"Due date {due_date.date()} is not after invoice date {inv_date.date()}"
            )

        # Past due date
        if due_date and due_date < now:
            report.add(
                "PAST_DUE_DATE",
                detail=f"Invoice was due on {due_date.date()} ({(now - due_date).days} days ago)"
            )

    def _check_unknown_buyer(self, report: AnomalyReport, buyer_id: Optional[str]) -> None:
        """Flag if buyer could not be matched to the known database."""
        if not buyer_id:
            report.add(
                "UNKNOWN_BUYER",
                detail="Buyer name did not match any registered buyer in the system."
            )

    def _check_confidence(self, report: AnomalyReport, fields) -> None:
        """Flag low overall extraction confidence."""
        if fields.overall_confidence < self.MIN_CONFIDENCE_THRESHOLD:
            report.add(
                "LOW_EXTRACTION_CONFIDENCE",
                detail=f"Overall extraction confidence: {fields.overall_confidence:.0%}"
            )

    # ── Risk Score ──────────────────────────────

    def _compute_risk_score(self, report: AnomalyReport) -> float:
        """
        Compute a 0–1 risk score from anomaly severity.
        High severity = 0.4 points, medium = 0.2, low = 0.1 (capped at 1.0)
        """
        severity_weights = {"high": 0.4, "medium": 0.2, "low": 0.1}
        score = sum(severity_weights.get(a.severity, 0) for a in report.anomalies)
        return min(score, 1.0)

    # ── History Management ──────────────────────

    def register_invoice(self, invoice_number: str, buyer_id: Optional[str],
                         amount: Optional[float], seller_name: Optional[str]) -> None:
        """Call this after successfully processing an invoice to update history."""
        self.history.add(invoice_number, buyer_id, amount, seller_name)
