# Simple-Diabetes-Risk-Predictor-2.0

Perfect — here’s a **clean, copy-pasteable README.md** you can directly drop into your GitHub repo:

````markdown
# Diabetes Prediction System

A machine learning project to predict diabetes risk using the **Pima Indians Diabetes Dataset**.  
Built with **PyTorch** (model), **FastAPI** (API), and **Streamlit** (UI).

## Features
- Neural network with PyTorch  
- FastAPI backend for predictions  
- Streamlit web interface  
- Data preprocessing & scaling  

## Installation

```bash
git clone <repository-url>
cd diabetes-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
````

## Usage

### Train Model

```bash
python train_torch.py
```

### Run Backend (FastAPI)

```bash
uvicorn main:app --reload
# Docs: http://localhost:8000/docs
```

### Run Frontend (Streamlit)

```bash
streamlit run streamlit_app.py
# UI: http://localhost:8501
```

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 120, 70, 20, 85, 25.0, 0.5, 30]}'
```

## Project Structure

```
diabetes-predictor/
├── main.py
├── train_torch.py
├── streamlit_app.py
├── models.py
├── requirements.txt
└── README.md
```

## License

MIT License

```

---

✅ This is short, professional, and **GitHub-ready**.  
Want me to also add **badges** (Python version, license, stars) at the top for a polished look?
```
