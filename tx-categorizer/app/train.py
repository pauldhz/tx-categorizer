#!/usr/bin/env python3
"""Train a first ML model to categorize bank transactions.

- Uses your labeled CSV (Category + _subcategory).
- Learns from Description + a few optional structured fields (Type, Montant).
- Saves a single sklearn Pipeline to models/tx_model.joblib.

Why a single Pipeline?
- It bundles preprocessing (TF-IDF, one-hot, scaling) + classifier.
- In production you just load the pipeline and call predict / predict_proba.

Usage:
  python train.py --data export_labeled_clean.csv --out models/tx_model.joblib --report models/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple
import matplotlib.pylab as plt

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = ["description", "_category", "_subcategory"]

import re
import unicodedata


def normalize_description(description: str) -> str:
    """
    Normalize a transaction description for rules / ML usage.

    Steps:
    - handle None
    - lowercase
    - remove accents
    - remove punctuation
    - normalize spaces
    """
    if not description:
        return ""

    # Lowercase
    text = description.lower()

    # Remove accents (é → e, ç → c, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Remove punctuation / special chars (keep letters & numbers)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def parse_amount(x) -> float:
    """Parse 'Montant' that may use comma decimals (e.g. '29,21')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Trim labels (super important)
    df["_category"] = df["_category"].fillna("").astype(str).str.strip()
    df["_subcategory"] = df["_subcategory"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).apply(normalize_description)

    # Keep only labeled rows
    df = df[(df["_category"] != "") & (df["_subcategory"] != "")].reset_index(drop=True)

    # Optional fields
    df["type"] = df["type"].fillna("").astype(str) if "type" in df.columns else ""
    df["MontantNum"] = df["amount"].apply(parse_amount) if "amount" in df.columns else 0.0

    # Target label = combined
    df["Label"] = df["_category"] + " / " + df["_subcategory"]
    return df


def build_pipeline() -> Pipeline:
    """
    Baseline model (fast):
    - TF-IDF on Description (bigrams)
    - SGDClassifier with log_loss (gives predict_proba)
    - Add a bit of structure (type, amount)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("desc_tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=12000,
            ), "description"),
            ("cat_onehot", OneHotEncoder(handle_unknown="ignore"), ["type"]),
            ("num", Pipeline(steps=[
                ("scaler", StandardScaler(with_mean=False)),
            ]), ["MontantNum"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


def safe_train_test_split(X, y, test_size=0.2, random_state=42) -> Tuple:
    """Try stratified split; if some labels are too small, fall back."""
    counts = pd.Series(y).value_counts()
    if (counts < 2).any():
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV (semicolon-separated)")
    parser.add_argument("--out", required=True, help="Path to save the trained model (.joblib)")
    parser.add_argument("--report", default=None, help="Optional path to save a JSON report")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    if len(df) < 30:
        raise ValueError(f"Not enough labeled rows to train reliably (found {len(df)}).")

    X = df[["description", "type", "MontantNum"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plt.scatter(y_pred, y_test)
    plt.show()
    report_txt = classification_report(y_test, y_pred, zero_division=0)

    print("\n=== Classification report (hold-out test set) ===\n")
    print(report_txt)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    report_data = {
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "labels": sorted(df["Label"].unique().tolist()),
        "classification_report_text": report_txt,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved model to: {out_path}\n")
    if args.report:
        print(f"Saved report to: {args.report}\n")


if __name__ == "__main__":
    main()
