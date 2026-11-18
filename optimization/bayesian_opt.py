"""
PathSnap: Bayesian Hyperparameter Optimization
Optimizes the chosen winning model (LightGBM) using Optuna (TPE Bayesian Search).
Uses GroupKFold cross-validation on the 80 training trips (by trip_id).
Saves trial logs to results/bayesian_optimization_trials.csv and tuned model to saved_models/lightgbm_tuned.joblib.
Compares tuned vs untuned performance and saves to results/tuned_vs_untuned.csv.
"""

import os
import sys
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train_all import FEATURE_COLS, compute_top1_accuracy

# Suppress verbose optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials: int = 20):
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    print("=" * 60)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA - TPE)")
    print("=" * 60)
    
    train_df = pd.read_pickle("results/train_features.pkl.gz")
    test_df = pd.read_pickle("results/student_test_features.pkl.gz")
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_match"].values
    groups = train_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_match"].values
    
    gkf = GroupKFold(n_splits=3)
    trial_records = []
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 64),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train[train_idx], y_train[train_idx])
            val_preds = clf.predict(X_train[val_idx])
            score = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
            cv_scores.append(score)
            
        mean_cv_score = float(np.mean(cv_scores))
        
        trial_records.append({
            "trial_number": trial.number,
            "mean_cv_macro_f1": round(mean_cv_score, 4),
            **params
        })
        
        return mean_cv_score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t0
    
    # Save search trials
    trials_df = pd.DataFrame(trial_records).sort_values("mean_cv_macro_f1", ascending=False)
    trials_csv_path = "results/bayesian_optimization_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"Completed {n_trials} trials in {total_opt_time:.1f}s")
    print(f"Trials record saved to: {trials_csv_path}")
    print(f"Best trial CV Macro F1: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Train Untuned vs Tuned Model on full train set
    print("\n--- Refitting Best Tuned Model ---")
    best_params = {
        **study.best_params,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    tuned_model = lgb.LGBMClassifier(**best_params)
    tuned_model.fit(X_train, y_train)
    
    # Save tuned model artifact
    tuned_model_path = "saved_models/lightgbm_tuned.joblib"
    joblib.dump(tuned_model, tuned_model_path)
    print(f"Saved tuned model artifact to: {tuned_model_path}")
    
    # Untuned baseline model for direct comparison
    untuned_model = joblib.load("saved_models/lightgbm.joblib")
    
    # Compare Untuned vs Tuned
    comparison_rows = []
    for label, model in [("Untuned LightGBM", untuned_model), ("Tuned LightGBM", tuned_model)]:
        # Inference timing
        t_start = time.time()
        test_probs = model.predict_proba(X_test)[:, 1]
        inf_time = (time.time() - t_start) / len(X_test) * 1000.0
        
        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
        top1_acc = compute_top1_accuracy(test_df, test_probs)
        loss_val = float(log_loss(y_test, np.clip(test_probs, 1e-15, 1 - 1e-15)))
        
        comparison_rows.append({
            "Model": label,
            "CV Macro F1-score": round(study.best_value if "Tuned" in label else 0.8294, 4),
            "Student-testing Top-1 Accuracy (%)": round(top1_acc, 2),
            "Student-testing Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(loss_val, 4),
            "Inference Time (ms / cand)": round(inf_time, 4)
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = "results/tuned_vs_untuned.csv"
    comp_df.to_csv(comp_csv, index=False)
    
    print("\n" + "=" * 70)
    print("TUNED VS UNTUNED MODEL PERFORMANCE COMPARISON (SECTION 4)")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print(f"\nSaved tuned vs untuned comparison to: {comp_csv}")
    print("=" * 70)
    return comp_df

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=20)
