# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import numpy as np
import torch
from models import DiabetesNet

app = FastAPI(title="Diabetes Predictor (PyTorch)")

# Pydantic schema: expects a list of floats in the same order as training FEATURE_NAMES
class PredictRequest(BaseModel):
    features: list[float]

# Load artifacts at startup
@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_names, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("scaler.pkl")
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
    model = DiabetesNet(input_dim=len(feature_names)).to(device)
    model.load_state_dict(torch.load("model_torch.pth", map_location=device))
    model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Validate
    if len(req.features) != len(feature_names):
        raise HTTPException(status_code=400, detail=f"features must be length {len(feature_names)} (order: {feature_names})")

    x = np.array(req.features, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)  # shape (1, n)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)         # shape (1,) or (,)
        prob = torch.sigmoid(logits).cpu().numpy().item()  # float
    label = 1 if prob >= 0.5 else 0
    return {"probability": prob, "label": label}
