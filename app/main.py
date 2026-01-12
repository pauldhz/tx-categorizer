from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_category

app = FastAPI(title="Transaction Categorizer")

class TxIn(BaseModel):
    date: str
    type: str
    description: str
    montant: float
    sens: str

@app.post("/predict")
def predict(tx: TxIn):
    result = predict_category(tx.model_dump())
    return result