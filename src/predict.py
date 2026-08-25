import json
from pathlib import Path

import joblib
import pandas as pd


def load_model_and_scaler(model_path='models/final_model.pkl', scaler_path='models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_input(input_path='input.json'):
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data


def predict(model, scaler, input_data, threshold=0.5):
    df = pd.DataFrame([input_data])
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


def save_output(result, output_path='output.json'):
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model, scaler = load_model_and_scaler(str(project_root / 'models/final_model.pkl'), str(project_root / 'models/scaler.pkl'))
    input_data = load_input(str(project_root / 'input.json'))
    result = predict(model, scaler, input_data)
    save_output(result, str(project_root / 'output.json'))
    print("Prediction saved to output.json")
    print(result)