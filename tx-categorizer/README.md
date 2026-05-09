# tx-categorizer
## Entrainement
```shell
python app/train.py --data training/data_training.csv --out app/models/tx_model.joblib --report app/models/report.json
```

## Démarrer l'app
```shell
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Test CURL
```shell
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-08-08",
    "type": "CARD_TRANSACTION",
    "description": "GRAND FRAIS AULNOY",
    "montant": -38.74
  }'
  ```

## CLI pour prédire un csv
```shell
python scripts/categorize_csv.py data/transactions.csv data/transactions_categorized.csv
```