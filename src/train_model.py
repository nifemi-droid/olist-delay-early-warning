import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# 1. Load Data
print("Loading feature table...")
df = pd.read_csv("data/features.csv")

# NEW: Convert purchase_time to datetime (matches v3 SQL output)
df["purchase_time"] = pd.to_datetime(df["purchase_time"])
df = df.sort_values("purchase_time") # CRITICAL: Sort by time!

# 2. Time-Based Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Train size: {len(train_df):,} orders (from {train_df['purchase_time'].min().date()} to {train_df['purchase_time'].max().date()})")
print(f"Test size:  {len(test_df):,} orders (from {test_df['purchase_time'].min().date()} to {test_df['purchase_time'].max().date()})")

# Drop columns we don't want the model to see
# NEW: match the v3 column names
drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["is_late"]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["is_late"]

# 3. Preprocessing Pipeline
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# 4. Models
models = {
    "Baseline (Never Late)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight="balanced")
}

fitted, rows = {}, []

for name, clf in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba)
    })

# 5. Show Results
print("\n--- Model Results (Time-Based Split) ---")
results_df = pd.DataFrame(rows).round(3)
print(results_df.to_string(index=False))

# 6. Confusion Matrix
print("\n--- Confusion Matrix (Logistic Regression) ---")
best_pipe = fitted["Logistic Regression"]
proba_lr = best_pipe.predict_proba(X_test)[:, 1]
pred_lr = (proba_lr >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_lr).ravel()

print(f"True Positives (Caught Delays):  {tp}")
print(f"False Negatives (Missed Delays): {fn}")
print(f"False Positives (Wasted Alerts): {fp}")
print(f"True Negatives (Correct On-Time): {tn}")

# Save the trained model
import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/olist_model.joblib")
print("Model saved to models/olist_model.joblib")