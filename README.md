# Simple-Diabetes-Risk-Predictor-2.0


A machine learning project to predict diabetes risk using the **Pima Indians Diabetes Dataset**.  
Built with **PyTorch** (model), **FastAPI** (API), and **Streamlit** (UI).

## Installation

```bash
git clone https://github.com/BhargavSawant/Simple-Diabetes-Risk-Predictor-2.0.git
```
```
cd diabetes-predictor
```
## Create virtual environment
```
python -m venv venv        # Creation of virtual environmenet
source venv/bin/activate   # Activation of virtual environment for Mac/Linux
venv\Scripts\activate      # Activation of virtual environment for Windows
```
```
# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
python train_torch.py
```

### Run Backend (FastAPI)

```bash
uvicorn main:app --reload
```

### Run Frontend (Streamlit)

```bash
streamlit run streamlit_app.py
```
