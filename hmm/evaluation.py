"""
PathSnap: HMM vs Classifier Evaluation on Student-Test Trips
Evaluates all 10 student testing trips.
Compares:
1. Pointwise Top-1 Accuracy (Classifier-Only vs Classifier + HMM)
2. Route Continuity (Classifier-Only vs Classifier + HMM)
3. Reconstructed Route vs Reference Route recovery
Saves results to results/hmm_vs_classifier.csv and generates comparison plots.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import load_trip
from features.candidate_extractor import CandidateExtractor
from hmm.viterbi import HMMMapMatcher

def evaluate_all_student_test_trips():
    os.makedirs("results", exist_ok=True)
    test_dir = "data/student_test"
    trip_ids = sorted(os.listdir(test_dir))
    
    print("=" * 60)
    print("RUNNING HMM ROUTE RECONSTRUCTION ON STUDENT-TEST TRIPS")
    print("=" * 60)
    
    # Load tuned LightGBM model
    model = joblib.load("saved_models/lightgbm_tuned.joblib")
    
    eval_rows = []
    
    for tid in trip_ids:
        t0 = time.time()
        trip = load_trip(test_dir, tid, load_route=True)
        extractor = CandidateExtractor(trip, top_k=8)
        matcher = HMMMapMatcher(trip, model)
        
        recon = matcher.run_reconstruction(extractor)
        duration = time.time() - t0
        
        # Ground truth route overlap
        pred_route = set(recon["reconstructed_route"])
        true_route = set(trip.route) if trip.route is not None else set()
        
        if len(true_route) > 0:
            precision = len(pred_route.intersection(true_route)) / max(len(pred_route), 1)
            recall = len(pred_route.intersection(true_route)) / max(len(true_route), 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        else:
            f1, precision, recall = 0.0, 0.0, 0.0
            
        print(f"Trip {tid}:")
        print(f"  Obs Accuracy:  Classifier={recon['observation_acc_clf']:.2f}%  -->  HMM={recon['observation_acc_hmm']:.2f}%")
        print(f"  Continuity:    Classifier={recon['continuity_clf']:.2f}%  -->  HMM={recon['continuity_hmm']:.2f}%")
        print(f"  Route Arcs:    Pred={len(recon['reconstructed_route'])}, True={len(true_route)}, F1={f1:.4f}")
        print(f"  Process Time:  {duration:.2f}s\n")
        
        eval_rows.append({
            "trip_id": tid,
            "gps_points": len(trip.track),
            "true_route_arcs": len(true_route),
            "pred_route_arcs": len(recon["reconstructed_route"]),
            "clf_pointwise_acc": recon["observation_acc_clf"],
            "hmm_pointwise_acc": recon["observation_acc_hmm"],
            "clf_continuity": recon["continuity_clf"],
            "hmm_continuity": recon["continuity_hmm"],
            "route_precision": round(precision * 100.0, 2),
            "route_recall": round(recall * 100.0, 2),
            "route_f1": round(f1, 4),
            "time_seconds": round(duration, 2)
        })
        
    df_results = pd.DataFrame(eval_rows)
    csv_path = "results/hmm_vs_classifier.csv"
    df_results.to_csv(csv_path, index=False)
    
    # Summary row
    avg_clf_acc = df_results["clf_pointwise_acc"].mean()
    avg_hmm_acc = df_results["hmm_pointwise_acc"].mean()
    avg_clf_cont = df_results["clf_continuity"].mean()
    avg_hmm_cont = df_results["hmm_continuity"].mean()
    avg_f1 = df_results["route_f1"].mean()
    
    print("=" * 70)
    print("OVERALL STUDENT-TEST HMM RECONSTRUCTION SUMMARY (SECTION 5)")
    print("=" * 70)
    print(f"Average Classifier Pointwise Accuracy: {avg_clf_acc:.2f}%")
    print(f"Average HMM Pointwise Accuracy:        {avg_hmm_acc:.2f}%  (+{avg_hmm_acc - avg_clf_acc:.2f}%)")
    print(f"Average Classifier Route Continuity:   {avg_clf_cont:.2f}%")
    print(f"Average HMM Route Continuity:          {avg_hmm_cont:.2f}%  (+{avg_hmm_cont - avg_clf_cont:.2f}%)")
    print(f"Average Sequence Route F1-score:       {avg_f1:.4f}")
    print(f"Detailed results saved to:             {csv_path}")
    print("=" * 70)
    
    # Generate HMM vs Classifier comparison bar chart
    plt.figure(figsize=(9, 5))
    metrics = ["Pointwise Accuracy (%)", "Route Continuity (%)"]
    clf_vals = [avg_clf_acc, avg_clf_cont]
    hmm_vals = [avg_hmm_acc, avg_hmm_cont]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, clf_vals, width, label="Classifier-Only", color="#7293cb")
    plt.bar(x + width/2, hmm_vals, width, label="Classifier + HMM/Viterbi", color="#2ca02c")
    
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title("Classifier-Only vs Classifier + HMM/Viterbi Decoding", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 110)
    
    for i in range(len(metrics)):
        plt.text(i - width/2, clf_vals[i] + 1.5, f"{clf_vals[i]:.1f}%", ha='center', fontweight='bold')
        plt.text(i + width/2, hmm_vals[i] + 1.5, f"{hmm_vals[i]:.1f}%", ha='center', fontweight='bold', color="#1b661b")
        
    plt.legend(frameon=True)
    plt.tight_layout()
    chart_path = "results/hmm_vs_classifier_chart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to: {chart_path}")
    
    return df_results

if __name__ == "__main__":
    evaluate_all_student_test_trips()
