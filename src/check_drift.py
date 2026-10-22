import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading data...")
df = pd.read_csv("data/features.csv")
df["purchase_time"] = pd.to_datetime(df["purchase_time"])
df = df.sort_values("purchase_time")

# Split into early (reference) and late (current)
split_idx = int(len(df) * 0.8)
reference_data = df.iloc[:split_idx].drop(columns=["order_id", "purchase_time", "est_delivery"])
current_data = df.iloc[split_idx:].drop(columns=["order_id", "purchase_time", "est_delivery"])

print("Generating Evidently Drift Report...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save as HTML
report.save_html("docs/data_drift_report.html")
print("Success! Open docs/data_drift_report.html in your browser to see the drift.")