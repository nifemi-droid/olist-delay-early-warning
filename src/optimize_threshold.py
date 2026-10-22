import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# 1. Load and split exactly as before
df = pd.read_csv("data/features.csv")

# UPDATED: Use the new column names from v3
df["purchase_time"] = pd.to_datetime(df["purchase_time"])
df = df.sort_values("purchase_time")
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# UPDATED: Drop the new v3 column names
drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["is_late"]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df["is_late"]

# 2. Train the Logistic Regression model
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]

# 3. Business Assumptions
COST_OF_DELAY = 50
COST_OF_FALSE_ALARM = 5

results = []
for t in np.arange(0.10, 0.91, 0.05):
    flagged = proba >= t
    tp = ((flagged) & (y_test == 1)).sum()
    fp = ((flagged) & (y_test == 0)).sum()
    fn = ((~flagged) & (y_test == 1)).sum()
    
    # Business Value = Money saved from catching delays - Money wasted on false alarms
    value = (tp * COST_OF_DELAY) - (fp * COST_OF_FALSE_ALARM)
    
    results.append({
        "threshold": round(t, 2), 
        "flagged": int(flagged.sum()),
        "caught_delays": int(tp), 
        "wasted_alerts": int(fp),
        "missed_delays": int(fn),
        "net_value": int(value)
    })

thr = pd.DataFrame(results)
best_row = thr.loc[thr["net_value"].idxmax()]
best_t = best_row["threshold"]
print(f"\nBest threshold under these assumptions: {best_t}")
print(thr.to_string(index=False))