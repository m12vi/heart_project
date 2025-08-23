"""
Train a heart disease prediction model and save as models/heart_model.pkl
"""
import os, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "data/heart.csv"
df = pd.read_csv(DATA_PATH)

target_col = "target"
feature_cols = [c for c in df.columns if c != target_col]
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = [c for c in feature_cols if c not in numeric_features]

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

candidates = {
    "log_reg": LogisticRegression(max_iter=2000),
    "rf": RandomForestClassifier(n_estimators=400, random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}
best_name, best_score = None, -1

for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    scores[name] = (float(np.mean(s)), [float(v) for v in s])
    if scores[name][0] > best_score:
        best_name, best_score = name, scores[name][0]

best_pipe = Pipeline([("prep", preprocess), ("model", candidates[best_name])])
best_pipe.fit(X_train, y_train)

y_proba = best_pipe.predict_proba(X_test)[:, 1] if hasattr(best_pipe.named_steps["model"], "predict_proba") else None
y_pred = best_pipe.predict(X_test)
test_roc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/heart_model.pkl")


with open("evaluation.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"CV ROC-AUC (mean): {scores[best_name][0]:.3f}\n")
    if test_roc is not None:
        f.write(f"Test ROC-AUC: {test_roc:.3f}\n")
    f.write("\nClassification report (test):\n")
    f.write(classification_report(y_test, y_pred))
print("Saved model to models/heart_model.pkl")


