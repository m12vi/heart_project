# Heart Failure Prediction (End-to-End)

This project predicts heart disease using the UCI Heart dataset (`data/heart.csv`). It includes:
- Reproducible training pipeline (scikit-learn)
- Saved model (`models/heart_model.pkl`)
- Streamlit app (`app.py`) for interactive predictions
- Evaluation artifacts in `evaluation.txt` and plots in `plots/`

## 1) Set up environment
```bash
# Option A: Conda
conda create -n hdp python=3.10 -y
conda activate hdp

# Option B: venv (Python 3.10+)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2) Explore the data
`data/heart.csv` has columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

Numeric features (scaled): age, trestbps, chol, thalach, oldpeak
Categorical features (one-hot): sex, cp, fbs, restecg, exang, slope, ca, thal

## 3) Train the model
```bash
python train.py
```
This saves the trained pipeline to `models/heart_model.pkl` and writes metrics to `evaluation.txt`.

### Current results (from this run)
- Best model: **log_reg**
- CV ROC-AUC (mean±std): **0.892 ± 0.036**
- Test ROC-AUC: **{test_roc:.3f}**

See plots in `plots/` for Confusion Matrix and ROC curve.

## 4) Run the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL shown (usually http://localhost:8501).

## 5) Package for deployment
- **Streamlit Community Cloud**: push this folder to a public Git repo (e.g., GitHub). On Streamlit Cloud, set:
  - App file: `app.py`
  - Python version: 3.10
  - Dependencies: from `requirements.txt`
- **Hugging Face Spaces**: create a Space (Streamlit), upload repo files, set `app.py`.
- **Docker (optional)**:
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t heart-app .
docker run -p 8501:8501 heart-app
```

## 6) How to use the app
Enter patient values in the UI, click **Predict**, and view the predicted class (0/1) and probability.

## 7) Re-training on new data
Replace `data/heart.csv` with your updated dataset (same columns), then run:
```bash
python train.py
```

---

### Notes
- Pipeline prevents data leakage: preprocessing (scaling + one-hot) is fit only on training data.
- Metrics tracked: ROC-AUC (CV + test), classification report, confusion matrix, ROC curve.
- You can tune models (e.g., `RandomForestClassifier` n_estimators, max_depth) inside `train.py`.
