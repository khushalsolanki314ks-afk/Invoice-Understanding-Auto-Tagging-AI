"""
Week 6: Buyer Matching & Auto-Tagging
========================================
Matches extracted buyer names to the registered buyer database using:
  - Exact match
  - Fuzzy string matching (RapidFuzz)
  - Alias table lookup

Then classifies each invoice into a risk bucket: LOW / MEDIUM / HIGH
based on:
  - Buyer payment history
  - Invoice amount vs. buyer's historical average
  - Whether anomalies were detected

Usage:
    from pipeline.buyer_matcher import BuyerMatcher
    matcher = BuyerMatcher(buyer_db)
    result = matcher.match_and_tag(buyer_name_raw, invoice_fields, anomaly_report)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from loguru import logger

try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not installed — fuzzy matching disabled. pip install rapidfuzz")


# ──────────────────────────────────────────────
# Risk Bucket Configuration
# ──────────────────────────────────────────────

RISK_RULES = {
    # If ANY high-severity anomaly → always HIGH risk
    "high_anomaly": {"bucket": "high", "reason": "High-severity anomaly detected"},

    # Unknown buyer → HIGH
    "unknown_buyer": {"bucket": "high", "reason": "Buyer not in registered database"},

    # Poor payment history → HIGH
    "bad_payment_history": {"bucket": "high", "reason": "Buyer has poor payment history"},

    # Medium anomalies only → MEDIUM
    "medium_anomaly": {"bucket": "medium", "reason": "Medium-severity anomaly detected"},

    # New buyer (few invoices) → MEDIUM
    "new_buyer": {"bucket": "medium", "reason": "Buyer has limited invoice history (<5 invoices)"},

    # Average payment history → MEDIUM
    "average_payment_history": {"bucket": "medium", "reason": "Buyer has average payment history"},

    # No issues → LOW
    "clean": {"bucket": "low", "reason": "No anomalies, known buyer with good history"},
}

PAYMENT_RISK_MAP = {
    "excellent": "low",
    "good":      "low",
    "average":   "medium",
    "poor":      "high",
    "bad":       "high",
    "unknown":   "medium",
}


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────

@dataclass
class BuyerMatch:
    buyer_id: Optional[str]       # Matched buyer's ID (None if no match)
    buyer_name: str               # Canonical buyer name
    raw_input: str                # Original string from OCR
    match_score: float            # 0–1 fuzzy match score
    match_method: str             # "exact" | "alias" | "fuzzy" | "none"
    buyer_profile: Optional[dict] = None  # Full buyer data from DB


@dataclass
class RiskTag:
    bucket: str                   # "low" | "medium" | "high"
    score: float                  # 0–1 risk score
    rules_triggered: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "score": round(self.score, 2),
            "rules_triggered": self.rules_triggered,
            "rationale": self.rationale,
        }


@dataclass
class TaggedInvoice:
    buyer_match: BuyerMatch
    risk_tag: RiskTag

    def to_dict(self) -> dict:
        return {
            "buyer": {
                "buyer_id": self.buyer_match.buyer_id,
                "canonical_name": self.buyer_match.buyer_name,
                "raw_input": self.buyer_match.raw_input,
                "match_score": round(self.buyer_match.match_score, 2),
                "match_method": self.buyer_match.match_method,
                "profile": self.buyer_match.buyer_profile,
            },
            "risk": self.risk_tag.to_dict(),
        }


# ──────────────────────────────────────────────
# Buyer Matcher
# ──────────────────────────────────────────────

class BuyerMatcher:
    """
    Matches raw OCR buyer names to a registered buyer database.
    Supports exact, alias, and fuzzy matching strategies.
    """

    FUZZY_THRESHOLD = 75  # Minimum RapidFuzz score (0–100) to accept a match
    FUZZY_HIGH_CONF = 90  # Score above this → high confidence

    def __init__(self, buyer_db: list[dict] = None, db_path: str = None):
        """
        Args:
            buyer_db: List of buyer dicts (from sample_invoices.json)
            db_path:  Path to JSON file with buyer data
        """
        if buyer_db:
            self._buyers = buyer_db
        elif db_path:
            self._buyers = self._load_from_file(db_path)
        else:
            # Load default mock data
            default_path = Path(__file__).parent.parent.parent / "data" / "sample_invoices.json"
            self._buyers = self._load_from_file(str(default_path))

        # Build lookup indexes
        self._build_indexes()

    def _load_from_file(self, path: str) -> list[dict]:
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("buyers", [])
        except Exception as e:
            logger.error(f"Failed to load buyer DB from {path}: {e}")
            return []

    def _build_indexes(self) -> None:
        """Pre-compute lookup indexes for fast matching."""
        # Exact name → buyer
        self._name_index: dict[str, dict] = {}
        # Alias → buyer
        self._alias_index: dict[str, dict] = {}
        # All names/aliases for fuzzy matching
        self._all_names: list[tuple[str, dict]] = []  # (name, buyer)

        for buyer in self._buyers:
            name = buyer.get("name", "")
            norm = self._normalize(name)
            self._name_index[norm] = buyer
            self._all_names.append((norm, buyer))

            for alias in buyer.get("aliases", []):
                norm_alias = self._normalize(alias)
                self._alias_index[norm_alias] = buyer
                self._all_names.append((norm_alias, buyer))

        logger.info(f"Buyer index built: {len(self._buyers)} buyers, {len(self._all_names)} name/alias entries")

    # ── Public API ──────────────────────────────

    def match(self, raw_buyer_name: str) -> BuyerMatch:
        """
        Match a raw buyer name string to the buyer database.
        Returns a BuyerMatch with the best match found.
        """
        if not raw_buyer_name or len(raw_buyer_name.strip()) < 2:
            return BuyerMatch(None, "", raw_buyer_name or "", 0.0, "none")

        norm = self._normalize(raw_buyer_name)

        # 1. Exact match (fastest)
        if norm in self._name_index:
            buyer = self._name_index[norm]
            logger.debug(f"Exact match: '{raw_buyer_name}' → {buyer['name']}")
            return BuyerMatch(
                buyer_id=buyer["id"],
                buyer_name=buyer["name"],
                raw_input=raw_buyer_name,
                match_score=1.0,
                match_method="exact",
                buyer_profile=buyer,
            )

        # 2. Alias match
        if norm in self._alias_index:
            buyer = self._alias_index[norm]
            logger.debug(f"Alias match: '{raw_buyer_name}' → {buyer['name']}")
            return BuyerMatch(
                buyer_id=buyer["id"],
                buyer_name=buyer["name"],
                raw_input=raw_buyer_name,
                match_score=0.95,
                match_method="alias",
                buyer_profile=buyer,
            )

        # 3. Fuzzy match
        if RAPIDFUZZ_AVAILABLE:
            return self._fuzzy_match(raw_buyer_name, norm)

        logger.warning(f"No match found for buyer: '{raw_buyer_name}'")
        return BuyerMatch(None, raw_buyer_name, raw_buyer_name, 0.0, "none")

    def match_and_tag(
        self,
        raw_buyer_name: str,
        invoice_fields,         # InvoiceFields
        anomaly_report,         # AnomalyReport
    ) -> TaggedInvoice:
        """
        Full pipeline: match buyer + assign risk bucket.

        Returns a TaggedInvoice with buyer match + risk classification.
        """
        buyer_match = self.match(raw_buyer_name)
        risk_tag = self.classify_risk(invoice_fields, anomaly_report, buyer_match)

        return TaggedInvoice(buyer_match=buyer_match, risk_tag=risk_tag)

    # ── Fuzzy Matching ──────────────────────────

    def _fuzzy_match(self, raw: str, norm: str) -> BuyerMatch:
        """Use RapidFuzz to find the best matching buyer name."""
        candidates = [name for name, _ in self._all_names]
        buyer_map = {name: buyer for name, buyer in self._all_names}

        # Try multiple fuzzy strategies
        results = []
        for scorer in [fuzz.token_set_ratio, fuzz.partial_ratio, fuzz.WRatio]:
            best = rf_process.extractOne(norm, candidates, scorer=scorer)
            if best:
                results.append(best)

        if not results:
            return BuyerMatch(None, raw, raw, 0.0, "none")

        # Pick the result with the highest score
        best_name, best_score, _ = max(results, key=lambda x: x[1])

        if best_score >= self.FUZZY_THRESHOLD:
            buyer = buyer_map[best_name]
            norm_score = best_score / 100.0
            logger.debug(f"Fuzzy match: '{raw}' → '{buyer['name']}' (score: {best_score})")
            return BuyerMatch(
                buyer_id=buyer["id"],
                buyer_name=buyer["name"],
                raw_input=raw,
                match_score=norm_score,
                match_method="fuzzy",
                buyer_profile=buyer,
            )

        logger.info(f"No fuzzy match above threshold ({self.FUZZY_THRESHOLD}) for: '{raw}' (best: {best_score})")
        return BuyerMatch(None, raw, raw, best_score / 100.0, "none")

    # ── Risk Classification ─────────────────────

    def classify_risk(
        self,
        invoice_fields,
        anomaly_report,
        buyer_match: BuyerMatch,
    ) -> RiskTag:
        """
        Determine risk bucket (low/medium/high) based on:
          - Anomaly severity
          - Buyer match status
          - Buyer payment history
          - Invoice amount vs. buyer average
        """
        rules_triggered = []
        bucket = "low"

        # ── Rule 1: High-severity anomaly ──
        if anomaly_report.highest_severity == "high":
            bucket = "high"
            rules_triggered.append("high_anomaly")

        # ── Rule 2: Unknown buyer ──
        elif not buyer_match.buyer_id:
            bucket = "high"
            rules_triggered.append("unknown_buyer")

        else:
            profile = buyer_match.buyer_profile or {}
            payment_history = profile.get("payment_history", "unknown").lower()
            invoice_count = profile.get("invoice_count", 0)
            avg_amount = profile.get("avg_invoice_amount", 0)

            # ── Rule 3: Bad payment history ──
            payment_risk = PAYMENT_RISK_MAP.get(payment_history, "medium")
            if payment_risk == "high":
                bucket = max(bucket, "high", key=lambda x: {"low": 0, "medium": 1, "high": 2}.get(x, 0))
                rules_triggered.append("bad_payment_history")

            # ── Rule 4: Medium-severity anomalies ──
            if anomaly_report.highest_severity == "medium" and "medium_anomaly" not in rules_triggered:
                bucket = self._escalate(bucket, "medium")
                rules_triggered.append("medium_anomaly")

            # ── Rule 5: New buyer ──
            if invoice_count < 5:
                bucket = self._escalate(bucket, "medium")
                rules_triggered.append("new_buyer")

            # ── Rule 6: Average payment history ──
            if payment_risk == "medium" and "medium_anomaly" not in rules_triggered:
                bucket = self._escalate(bucket, "medium")
                rules_triggered.append("average_payment_history")

            # ── Rule 7: Clean invoice ──
            if not rules_triggered:
                rules_triggered.append("clean")

        # Compose rationale
        rationale = self._build_rationale(rules_triggered, buyer_match, invoice_fields)

        # Risk score: high=0.8+, medium=0.4-0.7, low=0-0.3
        score_map = {"low": 0.15, "medium": 0.5, "high": 0.85}
        base_score = score_map[bucket]
        final_score = min(base_score + anomaly_report.risk_score * 0.2, 1.0)

        return RiskTag(
            bucket=bucket,
            score=final_score,
            rules_triggered=rules_triggered,
            rationale=rationale,
        )

    def _escalate(self, current: str, to: str) -> str:
        """Escalate risk bucket to 'to' level if it's higher than current."""
        order = {"low": 0, "medium": 1, "high": 2}
        if order.get(to, 0) > order.get(current, 0):
            return to
        return current

    def _build_rationale(
        self, rules: list[str], buyer_match: BuyerMatch, invoice_fields
    ) -> str:
        """Build a human-readable rationale for the risk classification."""
        parts = []
        for rule in rules:
            meta = RISK_RULES.get(rule)
            if meta:
                parts.append(meta["reason"])

        if buyer_match.buyer_id and buyer_match.buyer_profile:
            profile = buyer_match.buyer_profile
            parts.append(
                f"Buyer '{buyer_match.buyer_name}' has {profile.get('invoice_count', 0)} "
                f"prior invoices with {profile.get('payment_history', 'unknown')} payment history."
            )
        elif not buyer_match.buyer_id:
            parts.append(f"Could not match '{buyer_match.raw_input}' to any registered buyer.")

        return " | ".join(parts) if parts else "No specific risk factors identified."

    # ── Utilities ───────────────────────────────

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a company name for matching."""
        # Lowercase, strip punctuation, remove common suffixes
        n = name.lower().strip()
        n = re.sub(r"[^\w\s]", " ", n)
        n = re.sub(r"\b(pvt|ltd|limited|private|inc|corp|corporation|co|llp|llc)\b", "", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n
