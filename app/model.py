from __future__ import annotations

from typing import Any, Dict

from app.rules import predict_by_rules

# Put ml_model.py into app/ml_model.py and import it like below
from app.ml_model import TxModel

_ml = TxModel()
try:
    print("Loading ml")
    _ml.load()
except Exception:
    print("ml not trained yet")
    # Model not trained yet => rules-only mode
    _ml = None


def predict_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    """Rules first, then ML, then fallback."""
    description = str(tx.get("description", "") or "")

    rule_result = predict_by_rules(description)
    if rule_result is not None:
        category, subcategory, rule_id = rule_result
        return {
            "category": category,
            "subcategory": subcategory,
            "confidence": 1.0,
            "method": "rules",
            "rule_id": rule_id,
        }

    if _ml is not None:
        print ("prediction applying")
        pred = _ml.predict(tx)
        if pred is not None:
            return {
                "category": pred.category,
                "subcategory": pred.subcategory,
                "confidence": pred.confidence,
                "method": "ml",
                "rule_id": None,
            }

    return {
        "category": "UNKNOWN",
        "subcategory": None,
        "confidence": 0.0,
        "method": "fallback",
        "rule_id": None,
    }
