"""Place this file as: app/ml_model.py

Loads the sklearn pipeline trained by train.py and exposes:
- TxModel.predict(tx_dict) -> MlPrediction(category, subcategory, confidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "tx_model.joblib"

@dataclass(frozen=True)
class MlPrediction:
    category: str
    subcategory: str
    confidence: float


def _parse_amount(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


class TxModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline = None

    def load(self) -> None:

        if not self.model_path.exists():
            print("not loaded")
            raise FileNotFoundError(
                f"ML model not found at {self.model_path}. Train it first with train.py."
            )
        print("loaded loaded")
        self.pipeline = joblib.load(self.model_path)

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, tx: Dict[str, Any]) -> Optional[MlPrediction]:
        if not self.is_loaded():
            return None

        desc = str(tx.get("description", "") or "")
        typ = str(tx.get("type", "") or "")
        sens = str(tx.get("sens", "") or "")
        montant = _parse_amount(tx.get("montant", 0.0))

        X = pd.DataFrame([{
            "Description": desc,
            "Type": typ,
            "Sens": sens,
            "MontantNum": montant,
        }])

        label = self.pipeline.predict(X)[0]

        confidence = 0.0
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)[0]
            confidence = float(np.max(proba))

        if " / " in label:
            category, subcategory = label.split(" / ", 1)
        else:
            category, subcategory = label, ""

        return MlPrediction(category=category, subcategory=subcategory, confidence=confidence)
