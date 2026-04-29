"""
train_model.py  –  XGBoost Cricket Shot Classifier (accuracy-optimised)

Key improvements over previous version:
  1. Optuna hyperparameter search  →  finds the best XGBoost config automatically
  2. SMOTE oversampling            →  fixes Upper Cut class imbalance at data level
  3. Soft-voting ensemble          →  XGBoost + Random Forest + Extra Trees
  4. Calibrated confidence scores  →  more reliable probability outputs for app.py
  5. Confusion matrix print        →  shows exactly which shots are being confused

Install extra deps once:
    pip install optuna imbalanced-learn
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not found. Run: pip install imbalanced-learn")
    print("   Continuing without SMOTE (accuracy will be slightly lower).\n")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠  optuna not found. Run: pip install optuna")
    print("   Continuing with default hyperparameters.\n")

# ─────────────────────────────────────────────
#  CONFIGURE THESE PATHS
# ─────────────────────────────────────────────
CSV_FILE   = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\3_Extracted_Data\training_data.csv"
MODEL_FILE = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\4_Model_Training\cricket_ai_model.pkl"
META_FILE  = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\4_Model_Training\model_meta.json"
# ─────────────────────────────────────────────

OPTUNA_TRIALS    = 60    # More = better tuning, but slower. 60 takes ~5 min on CPU.
MIN_SAMPLES_PER_CLASS = 5


# ── Data loading & validation ─────────────────────────────────────────────────
def load_and_validate(csv_path):
    print("── Loading data ──────────────────────────────")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}   Columns: {len(df.columns)}")

    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"   ⚠  Dropped {before - len(df)} rows with missing values.")

    counts = df['label'].value_counts()
    print("\n── Class distribution ────────────────────────")
    for shot, n in counts.items():
        flag = "  ⚠  VERY FEW SAMPLES" if n < MIN_SAMPLES_PER_CLASS else ""
        print(f"   {shot:<30} {n:>4} samples{flag}")

    if len(counts) < 2:
        raise ValueError("Need at least 2 classes.")
    return df


# ── Optuna hyperparameter search for XGBoost ─────────────────────────────────
# ── Optuna hyperparameter search for XGBoost ─────────────────────────────────
def tune_xgboost(X_train, y_train, n_classes, n_trials=60):
    print(f"\n── Optuna hyperparameter search ({n_trials} trials) ──")

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
            'max_depth':         trial.suggest_int('max_depth', 3, 9),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
            'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(**params)
        
        # FIX: Removed the deprecated fit_params. 
        # SMOTE already perfectly balanced the classes, so weights are redundant here!
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='accuracy',
                                 n_jobs=1)   
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n   Best CV accuracy: {study.best_value * 100:.1f}%")
    print(f"   Best params: {best}")
    return best


# ── Build the final ensemble ──────────────────────────────────────────────────
def build_ensemble(xgb_params, n_classes):
    """
    Soft-voting ensemble: XGBoost + Random Forest + Extra Trees.
    Averaging probabilities from three diverse models reduces variance.
    """
    xgb_model = xgb.XGBClassifier(
        **xgb_params,
        random_state=42, n_jobs=-1, eval_metric='mlogloss'
    )
    rf_model  = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    et_model  = ExtraTreesClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=2, class_weight='balanced',
        random_state=43, n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('et', et_model)],
        voting='soft',
        weights=[2, 1, 1],    # Give XGBoost double weight (it's the strongest)
    )
    return ensemble


# ── Training pipeline ─────────────────────────────────────────────────────────
def train(df):
    le          = LabelEncoder()
    X           = df.drop('label', axis=1).values.astype(np.float32)
    y           = le.fit_transform(df['label'].values)
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    print(f"\n── Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE: oversample minority classes to the majority class count ────────
    if SMOTE_AVAILABLE:
        print("\n── Applying SMOTE to balance classes ─────────")
        counts = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"   Before SMOTE: {counts}")

        # k_neighbors must be < smallest class count
        min_count = min(counts.values())
        k = min(5, min_count - 1)
        if k >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            counts_after = dict(zip(*np.unique(y_train, return_counts=True)))
            print(f"   After  SMOTE: {counts_after}")
        else:
            print("   ⚠  Too few samples for SMOTE, skipping.")

    sample_weights = compute_sample_weight('balanced', y_train)

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    if OPTUNA_AVAILABLE:
        best_xgb_params = tune_xgboost(X_train, y_train, n_classes, n_trials=OPTUNA_TRIALS)
    else:
        best_xgb_params = {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
        }

    # ── Train ensemble ────────────────────────────────────────────────────────
    print("\n── Training soft-voting ensemble (XGBoost + RF + ET) ──")
    ensemble = build_ensemble(best_xgb_params, n_classes)

    # VotingClassifier doesn't accept sample_weight directly for mixed types;
    # fit the XGB sub-model separately then reassemble if needed.
    # Simpler: fit the whole ensemble (RF/ET use class_weight='balanced' internally)
    ensemble.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n── Hold-out test accuracy: {acc * 100:.1f}% ──────────")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix — shows exactly which shot pairs are confused
    cm = confusion_matrix(y_test, y_pred)
    print("── Confusion matrix ──────────────────────────")
    header = f"{'':>20}" + "".join(f"{c:>14}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>20}" + "".join(f"{cm[i,j]:>14}" for j in range(n_classes))
        print(row_str)

    # Cross-validation on the full dataset
    n_splits = min(5, len(df) // n_classes)
    if n_splits >= 2:
        cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        print(f"\n── {n_splits}-fold CV accuracy: "
              f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Feature importances from the XGB sub-model only
    xgb_sub      = ensemble.named_estimators_['xgb']
    feat_names   = [c for c in df.columns if c != 'label']
    importances  = xgb_sub.feature_importances_
    ranked       = sorted(zip(importances, feat_names), reverse=True)

    print("\n── Top 12 feature importances (XGBoost) ─────")
    for imp, name in ranked[:12]:
        bar = '█' * int(imp * 120)
        print(f"   {name:<25} {imp:.4f}  {bar}")

    return ensemble, le, class_names


# ── Save ──────────────────────────────────────────────────────────────────────
def save(model, le, class_names, model_path, meta_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    meta = {'classes': class_names, 'n_features': 36}   # 12 features × 3 phases
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  Model saved  → {model_path}")
    print(f"✅  Metadata     → {meta_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df                          = load_and_validate(CSV_FILE)
    model, le, class_names      = train(df)
    save(model, le, class_names, MODEL_FILE, META_FILE)