"""
PathSnap: Evaluation Visualizations
Generates publication-quality charts for Section 3.3:
1. CV Macro F1 comparison across all 5 models
2. Student-test Top-1 Accuracy & Macro F1 comparison
3. Log Loss comparison
4. Confusion matrix of the best model (LightGBM)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train_all import FEATURE_COLS

def generate_comparison_plots():
    os.makedirs("results", exist_ok=True)
    comparison_df = pd.read_csv("results/model_comparison.csv")
    
    # 1. Macro F1 & Top-1 Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df["CV Macro F1-score"], width, label="CV Macro F1", color="#3470a3")
    plt.bar(x + width/2, comparison_df["Student-Test Macro F1-score"], width, label="Student-Test Macro F1", color="#f28e2b")
    
    plt.ylabel("Macro F1-score", fontsize=12)
    plt.title("Cross-Validation vs Student-Test Macro F1-score Across Models", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, comparison_df["Model"], rotation=15, ha='right', fontsize=10)
    plt.ylim(0.75, 0.90)
    plt.legend(frameon=True)
    plt.tight_layout()
    f1_plot_path = "results/model_f1_comparison.png"
    plt.savefig(f1_plot_path, dpi=200)
    plt.close()
    
    # 2. Top-1 Accuracy and Route Continuity Plot
    plt.figure(figsize=(10, 5))
    plt.plot(comparison_df["Model"], comparison_df["Top-1 Accuracy"], marker='o', linewidth=2.5, markersize=8, color="#2ca02c", label="Top-1 Accuracy (%)")
    plt.plot(comparison_df["Model"], comparison_df["Route continuity"], marker='s', linewidth=2.5, markersize=8, color="#d62728", label="Route Continuity (%)")
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title("Student-Test Top-1 Accuracy & Route Continuity Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    acc_plot_path = "results/model_accuracy_continuity.png"
    plt.savefig(acc_plot_path, dpi=200)
    plt.close()
    
    # 3. Log Loss Comparison
    plt.figure(figsize=(8, 4.5))
    valid_loss_df = comparison_df[comparison_df["Log Loss"] != "N/A"].copy()
    valid_loss_df["Log Loss"] = valid_loss_df["Log Loss"].astype(float)
    colors = ["#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    plt.bar(valid_loss_df["Model"], valid_loss_df["Log Loss"], color=colors[:len(valid_loss_df)], width=0.5)
    plt.ylabel("Log Loss (Lower is Better)", fontsize=12)
    plt.title("Candidate Classification Log Loss Comparison", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    loss_plot_path = "results/model_log_loss_comparison.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    
    # 4. Confusion Matrix for Best Model (LightGBM)
    test_df = pd.read_pickle("results/student_test_features.pkl.gz")
    best_model = joblib.load("saved_models/lightgbm.joblib")
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_match"].values
    
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Negative (0)", "Positive (1)"],
                yticklabels=["Negative (0)", "Positive (1)"])
    plt.title("LightGBM Confusion Matrix (Student-Test Set)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Candidate Class", fontsize=11)
    plt.ylabel("True Candidate Class", fontsize=11)
    plt.tight_layout()
    cm_plot_path = "results/confusion_matrix_lightgbm.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    
    print(f"Generated evaluation plots in results/:")
    print(f"  - {f1_plot_path}")
    print(f"  - {acc_plot_path}")
    print(f"  - {loss_plot_path}")
    print(f"  - {cm_plot_path}")

if __name__ == "__main__":
    generate_comparison_plots()
