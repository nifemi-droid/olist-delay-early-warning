import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="Olist Delay Risk API")

# Load the model we saved earlier
model = joblib.load("models/olist_model.joblib") 

# Use the business-optimal threshold we found earlier
THRESHOLD = 0.60 

class OrderRequest(BaseModel):
    features: Dict[str, float | str] 

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: OrderRequest):
    try:
        # Convert the incoming dictionary to a 1-row dataframe
        row = pd.DataFrame([req.features])
        
        # The model pipeline handles the preprocessing automatically
        risk = float(model.predict_proba(row)[:, 1][0])
        
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not score this input: {e}")
        
    return {
        "delay_risk": round(risk, 4),
        "flagged": bool(risk >= THRESHOLD),
        "threshold": THRESHOLD
    }