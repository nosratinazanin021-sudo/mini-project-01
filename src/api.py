from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

project_root = Path(__file__).parent.parent
model = joblib.load(project_root / 'models/final_model.pkl')
scaler = joblib.load(project_root / 'models/scaler.pkl')


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(transaction: Transaction, threshold: float = 0.5):
    df = pd.DataFrame([transaction.model_dump()])
    df_scaled = scaler.transform(df)

    probability = model.predict_proba(df_scaled)[:, 1][0]
    class_id = int(probability > threshold)
    prediction = "Fraud" if class_id == 1 else "Legitimate"

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability": round(float(probability), 4),
        "threshold": threshold,
        "status": "success"
    }