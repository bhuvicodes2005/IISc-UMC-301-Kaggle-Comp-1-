# Kaggle Competition 1 Solution

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# ================ CONFIG ================
FAST_RUN = True   # Set to False for full Kaggle run
N_FOLDS = 3 if FAST_RUN else 5
RANDOM_STATE = 42

TRAIN_PATH = "/kaggle/input/iisc-umc-301-kaggle-competition-1/train.csv"
TEST_PATH = "/kaggle/input/iisc-umc-301-kaggle-competition-1/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/iisc-umc-301-kaggle-competition-1/sample_submission.csv"

# ================ LOAD DATA ================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

TARGET = "song_popularity"
ID = "id"

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ================ FEATURES ================
features = [c for c in train.columns if c not in [TARGET, ID]]

# Simple feature engineering
train["duration_min"] = train["song_duration_ms"] / 60000
test["duration_min"] = test["song_duration_ms"] / 60000

# Log transforms
for col in ["tempo", "song_duration_ms", "loudness"]:
    for df in [train, test]:
        df[f"log_{col}"] = np.log1p(np.abs(df[col]))

# Squared terms
for col in ["energy", "audio_valence", "danceability"]:
    for df in [train, test]:
        df[f"{col}_squared"] = df[col] ** 2

# Interaction ratios
for df in [train, test]:
    df["speech_instrumental"] = df["speechiness"] / (1e-6 + df["instrumentalness"])
    df["dance_energy"] = df["danceability"] * df["energy"]
    df["valence_energy"] = df["audio_valence"] * df["energy"]

features = [c for c in train.columns if c not in [TARGET, ID]]

# ================ PREPROCESS ================
X = train[features]
y = train[TARGET]
X_test = test[features]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# ================ CV SETUP ================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ================ TRAIN MODELS ================
models = {
    "lgb": lgb.LGBMClassifier(n_estimators=500, random_state=RANDOM_STATE, verbose=-1),
    "cat": cb.CatBoostClassifier(iterations=500, random_seed=RANDOM_STATE, verbose=0),
    "xgb": xgb.XGBClassifier(n_estimators=500, eval_metric="auc", use_label_encoder=False, random_state=RANDOM_STATE),
    "lr": LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
}

# OOF predictions
oof_preds = np.zeros((len(X), len(models)))
test_preds = np.zeros((len(X_test), len(models)))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    X_tr_scaled, X_val_scaled = X_scaled[train_idx], X_scaled[valid_idx]

    for m_idx, (name, model) in enumerate(models.items()):
        if name == "lr":
            model.fit(X_tr_scaled, y_tr)
            oof_preds[valid_idx, m_idx] = model.predict_proba(X_val_scaled)[:, 1]
            test_preds[:, m_idx] += model.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
        else:
            model.fit(X_tr, y_tr)
            oof_preds[valid_idx, m_idx] = model.predict_proba(X_val)[:, 1]
            test_preds[:, m_idx] += model.predict_proba(X_test)[:, 1] / N_FOLDS

# Base model scores
for i, name in enumerate(models.keys()):
    auc = roc_auc_score(y, oof_preds[:, i])
    print(f"{name} OOF AUC: {auc:.4f}")

# ================ STACKING ================
meta = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
meta.fit(oof_preds, y)
final_preds = meta.predict_proba(test_preds)[:, 1]

final_auc = roc_auc_score(y, meta.predict_proba(oof_preds)[:, 1])
print("Stacked model OOF AUC:", final_auc)

# ================ SUBMISSION ================
submission = sample_sub.copy()
submission[TARGET] = final_preds
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")








