# Simple-Diabetes-Risk-Predictor-2.0


A machine learning project to predict diabetes risk using the **Pima Indians Diabetes Dataset**.  
Built with **PyTorch** (model), **FastAPI** (API), and **Streamlit** (UI).

# Installation

```bash
git clone <repository-url>
cd diabetes-predictor
```
# Create virtual environment
````markdown
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
