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
### Ouput 
<img width="1092" height="813" alt="Screenshot 2025-11-12 202126" src="https://github.com/user-attachments/assets/dbef4cf6-0a76-48cc-8ba3-11143201c71c" />
<img width="1251" height="905" alt="Screenshot 2025-11-12 202225" src="https://github.com/user-attachments/assets/2bffe90c-a2b9-42c4-8610-359014876096" />


