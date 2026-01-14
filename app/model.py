from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from app.rules import predict_by_rules


def predict_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point used by the API.

    Strategy (for now):
    1) Rules (deterministic)
    2) Fallback UNKNOWN

    Later we will add:
    - ML model (TF-IDF + LogisticRegression)
    - confidence thresholding
    """
    description = str(tx.get("description", "") or "")

    rule_result = predict_by_rules(description)
    if rule_result is not None:
        category, subcategory, rule_id = rule_result
        return {
            "category": category,
            "subcategory": subcategory,
            "confidence": 1.0,          # rules = deterministic
            "method": "rules",
            "rule_id": rule_id,
        }

    # fallback (no rule matched)
    return {
        "category": "UNKNOWN",
        "subcategory": None,
        "confidence": 0.0,
        "method": "fallback",
        "rule_id": None,
    }