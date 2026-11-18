"""
PathSnap: Multi-Model Experimentation & Comparison
Trains and evaluates 5 distinct models from different algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-Based)
3. Random Forest (Ensemble)
4. LightGBM (Boosting)
5. Multi-Layer Perceptron (Neural Network)

Evaluates GroupKFold CV (grouped by trip_id), Top-1 Accuracy, Macro F1,
Route Continuity, Log Loss, and Inference Time.
Saves comparison table to results/model_comparison.csv and trained artifacts to saved_models/.
"""

import os
import sys
import time
import joblib
import json
import numpy as np
import pandas as pd

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.pipeline import Pipeline
import lightgbm as lgb

FEATURE_COLS = [
    "candidate_rank",
    "dist_to_segment",
    "heading_diff",
    "dist_to_start",
    "dist_to_end",
    "proj_ratio",
    "segment_length",
    "gps_speed"
]

def compute_route_continuity(test_df: pd.DataFrame, preds_proba: np.ndarray) -> float:
    """
    Computes route continuity across test trips:
    Percentage of consecutive GPS observations whose top predicted road segments connect.
    """
    test_df = test_df.copy()
    test_df["pred_score"] = preds_proba
    
    # For each GPS observation, pick top candidate
    top_cands = test_df.sort_values(["trip_id", "point_idx", "pred_score"], ascending=[True, True, False])
    top_cands = top_cands.groupby(["trip_id", "point_idx"]).first().reset_index()
    
    total_transitions = 0
    connected_transitions = 0
    
    for _, trip_group in top_cands.groupby("trip_id"):
        trip_group = trip_group.sort_values("point_idx")
        arcs = trip_group["arc_id"].values
        from_nodes = trip_group["from_node"].values
        to_nodes = trip_group["to_node"].values
        
        for i in range(len(arcs) - 1):
            total_transitions += 1
            if arcs[i] == arcs[i+1]:
                connected_transitions += 1  # same segment
            elif to_nodes[i] == from_nodes[i+1]:
                connected_transitions += 1  # consecutive graph edge
                
    return (connected_transitions / max(total_transitions, 1)) * 100.0

def compute_top1_accuracy(test_df: pd.DataFrame, preds_proba: np.ndarray) -> float:
    """Computes Top-1 Accuracy: does the candidate with highest predicted score match ground truth?"""
    test_df = test_df.copy()
    test_df["pred_score"] = preds_proba
    top_cands = test_df.sort_values(["trip_id", "point_idx", "pred_score"], ascending=[True, True, False])
    top_cands = top_cands.groupby(["trip_id", "point_idx"]).first().reset_index()
    return float(top_cands["is_match"].mean() * 100.0)

def train_and_evaluate_all():
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    print("=" * 60)
    print("LOADING FEATURE MATRICES")
    print("=" * 60)
    
    train_df = pd.read_pickle("results/train_features.pkl.gz")
    test_df = pd.read_pickle("results/student_test_features.pkl.gz")
    
    X_train_full = train_df[FEATURE_COLS].values
    y_train_full = train_df["is_match"].values
    groups_train = train_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_match"].values
    
    print(f"Train samples: {len(X_train_full)} ({len(np.unique(groups_train))} trips)")
    print(f"Test samples:  {len(X_test)} ({len(np.unique(test_df['trip_id']))} trips)")
    
    # Subsample for faster CV on heavier models if needed
    sub_mask = np.random.RandomState(42).choice(len(train_df), size=min(100000, len(train_df)), replace=False)
    X_train_sub = X_train_full[sub_mask]
    y_train_sub = y_train_full[sub_mask]
    groups_sub = groups_train[sub_mask]
    
    # Model Definitions across 5 algorithm families
    models = {
        "Logistic Regression": {
            "family": "Linear and Probabilistic",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
            ]),
            "use_sub": False,
            "key_config": "C=1.0, solver=lbfgs, StandardScaler"
        },
        "KNN (Distance-Weighted)": {
            "family": "Instance-Based",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1))
            ]),
            "use_sub": True,
            "key_config": "k=9, weights=distance, StandardScaler"
        },
        "Random Forest": {
            "family": "Ensemble",
            "pipeline": RandomForestClassifier(
                n_estimators=100, max_depth=14, min_samples_split=10, random_state=42, n_jobs=-1
            ),
            "use_sub": False,
            "key_config": "n_estimators=100, max_depth=14"
        },
        "LightGBM": {
            "family": "Boosting",
            "pipeline": lgb.LGBMClassifier(
                n_estimators=120, learning_rate=0.08, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
            ),
            "use_sub": False,
            "key_config": "n_estimators=120, lr=0.08, num_leaves=31"
        },
        "Multi-Layer Perceptron (MLP)": {
            "family": "Neural Networks",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), activation="relu", max_iter=25,
                    early_stopping=True, random_state=42
                ))
            ]),
            "use_sub": True,
            "key_config": "layers=(64, 32), relu, Adam, early_stopping"
        }
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print("TRAINING AND COMPARING 5 ALGORITHM FAMILIES")
    print("=" * 60)
    
    gkf = GroupKFold(n_splits=5)
    
    for model_name, config in models.items():
        print(f"\n--- Training {model_name} ({config['family']}) ---")
        pipeline = config["pipeline"]
        
        # 1. 5-Fold GroupKFold Cross-Validation
        cv_f1s = []
        X_cv = X_train_sub if config["use_sub"] else X_train_full
        y_cv = y_train_sub if config["use_sub"] else y_train_full
        g_cv = groups_sub if config["use_sub"] else groups_train
        
        cv_t0 = time.time()
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_cv, y_cv, g_cv)):
            X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
            X_va, y_va = X_cv[val_idx], y_cv[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            val_preds = pipeline.predict(X_va)
            f1 = f1_score(y_va, val_preds, average="macro", zero_division=0)
            cv_f1s.append(f1)
            
        cv_macro_f1 = float(np.mean(cv_f1s))
        cv_time = time.time() - cv_t0
        print(f"  Group CV Macro F1: {cv_macro_f1:.4f} (5 folds in {cv_time:.1f}s)")
        
        # 2. Refit on full training set
        t_fit_start = time.time()
        pipeline.fit(X_cv, y_cv)
        train_time = time.time() - t_fit_start
        
        # 3. Test Evaluation on Student-Test
        t_inf_start = time.time()
        if hasattr(pipeline, "predict_proba"):
            test_probs = pipeline.predict_proba(X_test)[:, 1]
        else:
            test_probs = pipeline.predict(X_test)
        inf_time = (time.time() - t_inf_start) / len(X_test) * 1000.0  # ms per candidate
        
        test_preds = (test_probs >= 0.5).astype(int)
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
        top1_acc = compute_top1_accuracy(test_df, test_probs)
        continuity = compute_route_continuity(test_df, test_probs)
        
        try:
            # Clip probabilities to prevent infinite log loss
            clipped_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
            loss_val = float(log_loss(y_test, clipped_probs))
        except Exception:
            loss_val = np.nan
            
        print(f"  Top-1 Accuracy:     {top1_acc:.2f}%")
        print(f"  Route Continuity:   {continuity:.2f}%")
        print(f"  Student Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Log Loss:           {loss_val:.4f}")
        print(f"  Inference Time:     {inf_time:.3f} ms / candidate")
        
        # Save model artifact
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_save_path = f"saved_models/{safe_name}.joblib"
        joblib.dump(pipeline, model_save_path)
        
        results.append({
            "Model": model_name,
            "Algorithm Family": config["family"],
            "Key Configuration": config["key_config"],
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": round(top1_acc, 2),
            "Route continuity": round(continuity, 2),
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(loss_val, 4) if not np.isnan(loss_val) else "N/A",
            "Inference Time": f"{inf_time:.3f} ms"
        })
        
    summary_df = pd.DataFrame(results)
    table_csv = "results/model_comparison.csv"
    summary_df.to_csv(table_csv, index=False)
    
    print("\n" + "=" * 80)
    print("CONSOLIDATED MODEL COMPARISON TABLE (SECTION 3.1)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison table to: {table_csv}")
    print("=" * 80)
    return summary_df

if __name__ == "__main__":
    train_and_evaluate_all()
