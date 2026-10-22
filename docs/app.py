import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
import streamlit.components.v1 as components

# 1. Page Config
st.set_page_config(page_title="Olist Delay Command Center", page_icon="🚚", layout="wide")
st.title("🚚 Olist Order Delay Command Center")
st.markdown("An end-to-end dashboard for Operations, Data Science, and Business stakeholders.")

# 2. Load Data & Train Model (Cached for speed)
@st.cache_data
def load_and_split_data():
    df = pd.read_csv("data/features.csv")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df = df.sort_values("purchase_time")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

@st.cache_resource
def train_model(train_df):
    drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]
    X = train_df.drop(columns=drop_cols)
    y = train_df["is_late"]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(X, y)
    return pipe

with st.spinner("Training model and loading data..."):
    train_df, test_df = load_and_split_data()
    model = train_model(train_df)
    
    # Pre-calculate predictions on test set
    drop_cols = ["order_id", "purchase_time", "est_delivery", "is_late"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["is_late"]
    proba = model.predict_proba(X_test)[:, 1]

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Business Impact (Execs)", "🎯 Model Tuning (Data Team)", "🩺 Data Drift (Monitoring)"])

# ==========================================
# TAB 1: BUSINESS IMPACT
# ==========================================
with tab1:
    st.header("Campaign Profitability")
    st.markdown("Adjust the business assumptions to see how the model's ROI changes.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider("Risk Threshold", 0.10, 0.90, 0.60, 0.05)
    with col2:
        cost_delay = st.number_input("Cost of Missed Delay (R$)", value=50, step=10)
    with col3:
        cost_alarm = st.number_input("Cost of False Alarm (R$)", value=5, step=1)
        
    # Calculate curve for plot
    thresholds = np.arange(0.10, 0.91, 0.05)
    net_values = []
    for t in thresholds:
        flagged = proba >= t
        tp = ((flagged) & (y_test == 1)).sum()
        fp = ((flagged) & (y_test == 0)).sum()
        value = (tp * cost_delay) - (fp * cost_alarm)
        net_values.append(value)
        
    # Plotly Line Chart for Net Value
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=net_values, mode='lines+markers', name='Net Value', line=dict(color='green', width=3)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current Threshold")
    fig.update_layout(title="Net Value vs. Threshold", xaxis_title="Threshold", yaxis_title="Net Value (R$)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics at current threshold
    flagged = proba >= threshold
    tp = ((flagged) & (y_test == 1)).sum()
    fp = ((flagged) & (y_test == 0)).sum()
    current_value = (tp * cost_delay) - (fp * cost_alarm)
    
    st.info(f"At a threshold of **{threshold}**, the estimated campaign value is **R$ {current_value:,}** (Catching {tp} delays, generating {fp} false alarms).")

# ==========================================
# TAB 2: MODEL TUNING
# ==========================================
with tab2:
    st.header("Model Performance Tuning")
    st.markdown("Visualizing the trade-off between Precision and Recall.")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # 2. Precision/Recall vs. Threshold
    thresholds = np.arange(0.10, 0.91, 0.05)
    precisions = []
    recalls = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precisions.append(precision_score(y_test, pred, zero_division=0))
        recalls.append(recall_score(y_test, pred, zero_division=0))
        
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision', line=dict(color='blue')))
    fig_pr.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', name='Recall', line=dict(color='orange')))
    fig_pr.update_layout(title="Precision vs. Recall Trade-off", xaxis_title="Threshold", yaxis_title="Score")
    st.plotly_chart(fig_pr, use_container_width=True)

# ==========================================
# TAB 3: DATA DRIFT
# ==========================================
with tab3:
    st.header("Data Drift & Model Health")
    st.markdown("""
    This report compares the data the model was trained on (Early 2016–2018) versus the data it is currently seeing (Late 2018). 
    If features drift heavily, the model's predictions may become unreliable.
    """)
    
    drift_file = "docs/data_drift_report.html"
    if os.path.exists(drift_file):
        with open(drift_file, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=800, scrolling=True)
    else:
        st.warning("Drift report not found. Please run `python src/check_drift.py` first.")