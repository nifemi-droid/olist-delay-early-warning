import joblib
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def test_model_loads():
    """Quality Gate 1: The model must exist and load."""
    model = joblib.load("models/olist_model.joblib")
    assert model is not None

def test_predictions_are_probabilities():
    """Quality Gate 2: Predictions must be between 0 and 1."""
    model = joblib.load("models/olist_model.joblib")
    df = pd.read_csv("data/features.csv").head(10)
    drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]
    X = df.drop(columns=drop_cols)
    probas = model.predict_proba(X)[:, 1]
    assert all(probas >= 0) and all(probas <= 1)

def test_minimum_auc():
    """Quality Gate 3: The model must perform better than random (AUC > 0.70)."""
    model = joblib.load("models/olist_model.joblib")
    df = pd.read_csv("data/features.csv").sample(1000, random_state=42) # Fast test
    drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]
    X = df.drop(columns=drop_cols)
    y = df["is_late"]
    probas = model.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y, probas)
    assert auc > 0.70, f"Model quality dropped! AUC is {auc}"