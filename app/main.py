from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import predict_transaction

app = FastAPI(title="Transaction Categorizer", version="0.1.0")


class TxIn(BaseModel):
    date: str = Field(..., description="Date string from CSV, e.g. 01/10/2025")
    type: str = Field(..., description="Transaction type from CSV")
    description: str = Field(..., description="Bank description / merchant label")
    montant: float = Field(..., description="Amount (positive number)")
    sens: str = Field(..., description="DEBIT or CREDIT")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: TxIn):
    """
    Predict category/subcategory for a transaction.

    Returns:
      - category
      - subcategory
      - confidence (0..1)
      - method ("rules" | "fallback")
      - rule_id (optional)
    """
    return predict_transaction(tx.model_dump())