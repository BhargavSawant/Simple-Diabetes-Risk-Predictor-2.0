# train_torch.py
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models import DiabetesNet

# ---- Config ----
DATA_PATH = "pima_diabetes.csv"
FEATURE_NAMES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]
TARGET_COL = "Outcome"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------

# 1) Load & basic cleaning (replace zeroes for biologically invalid features)
df = pd.read_csv(DATA_PATH)
# Replace zeros with median for certain columns (common PIMA preprocessing)
cols_to_fix = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in cols_to_fix:
    if c in df.columns:
        df[c] = df[c].replace(0, np.nan)
        df[c].fillna(df[c].median(), inplace=True)

X = df[FEATURE_NAMES].values
y = df[TARGET_COL].values.astype(np.float32)

# 2) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4) Convert to torch tensors & DataLoader
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# 5) Model, loss, optimizer
model = DiabetesNet(input_dim=len(FEATURE_NAMES)).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()   # works with raw logits
optimizer = optim.Adam(model.parameters(), lr=LR)

# 6) Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)             # shape (batch,)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_ds)

    # optional quick val check every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logits_val = model(X_test_t.to(DEVICE))
            probs = torch.sigmoid(logits_val).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)
            roc = roc_auc_score(y_test, probs)
        print(f"Epoch {epoch}/{EPOCHS} — loss: {epoch_loss:.4f} val_acc: {acc:.4f} val_auc: {roc:.4f}")

# 7) Final evaluation
model.eval()
with torch.no_grad():
    logits_test = model(X_test_t.to(DEVICE))
    probs_test = torch.sigmoid(logits_test).cpu().numpy()
    preds = (probs_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    roc = roc_auc_score(y_test, probs_test)
print("FINAL EVAL — Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, AUC: {:.4f}".format(acc, prec, rec, roc))

# 8) Save artifacts
torch.save(model.state_dict(), "model_torch.pth")
joblib.dump(scaler, "scaler.pkl")
with open("feature_names.json", "w") as f:
    json.dump(FEATURE_NAMES, f)

print("Saved: model_torch.pth, scaler.pkl, feature_names.json")
