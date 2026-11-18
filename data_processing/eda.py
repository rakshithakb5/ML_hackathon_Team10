"""
PathSnap: Comprehensive Exploratory Data Analysis (EDA)
Inspects the 80 training trips and 10 student-testing trips.
Computes dataset inventory, trajectory stats, time gaps, graph connectivity,
and generates summary CSVs, JSON, and diagnostic plots.
"""

import os
import sys
import json
import glob

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from data_processing.loader import load_trip

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_eda(data_root: str = "data", results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    
    partitions = {
        "train": os.path.join(data_root, "train"),
        "student_test": os.path.join(data_root, "student_test")
    }
    
    inventory_rows = []
    sampling_deltas = []
    trip_lengths = []
    route_lengths = []
    arc_counts = []
    node_counts = []
    missing_values_count = 0
    non_monotonic_timestamps = 0
    
    print("=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    for partition_name, p_dir in partitions.items():
        if not os.path.exists(p_dir):
            print(f"Warning: Directory {p_dir} does not exist.")
            continue
            
        trip_ids = sorted(os.listdir(p_dir))
        print(f"Processing partition: {partition_name} ({len(trip_ids)} trips)...")
        
        for tid in trip_ids:
            trip_path = os.path.join(p_dir, tid)
            if not os.path.isdir(trip_path):
                continue
                
            try:
                trip = load_trip(p_dir, tid, load_route=True)
            except Exception as e:
                print(f"Error loading trip {tid}: {e}")
                continue
                
            n_pts = len(trip.track)
            n_nodes = len(trip.nodes)
            n_arcs = len(trip.arcs)
            n_route = len(trip.route) if trip.route is not None else 0
            
            # Check missing values / NaNs
            nan_track = np.isnan(trip.track).sum()
            nan_nodes = np.isnan(trip.nodes).sum()
            nan_arcs = np.isnan(trip.arcs).sum()
            total_nans = int(nan_track + nan_nodes + nan_arcs)
            missing_values_count += total_nans
            
            # Check timestamps
            ts = trip.track[:, 2]
            dts = np.diff(ts)
            if np.any(dts <= 0):
                non_monotonic_timestamps += 1
            sampling_deltas.extend(dts.tolist())
            
            trip_lengths.append(n_pts)
            if n_route > 0:
                route_lengths.append(n_route)
            arc_counts.append(n_arcs)
            node_counts.append(n_nodes)
            
            # Trajectory spatial extent
            lon_min, lat_min = np.min(trip.track[:, :2], axis=0)
            lon_max, lat_max = np.max(trip.track[:, :2], axis=0)
            
            # Trajectory duration and approx distance
            duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            track_m = trip.get_track_meters()
            step_dists = np.sqrt(np.sum(np.diff(track_m, axis=0)**2, axis=1))
            total_dist_m = float(np.sum(step_dists))
            avg_speed_kmh = float((total_dist_m / max(duration_s, 1.0)) * 3.6)
            
            inventory_rows.append({
                "partition": partition_name,
                "trip_id": tid,
                "gps_points": n_pts,
                "route_arcs": n_route,
                "network_nodes": n_nodes,
                "network_arcs": n_arcs,
                "duration_s": round(duration_s, 1),
                "distance_km": round(total_dist_m / 1000.0, 2),
                "avg_speed_kmh": round(avg_speed_kmh, 1),
                "lon_min": round(lon_min, 6),
                "lon_max": round(lon_max, 6),
                "lat_min": round(lat_min, 6),
                "lat_max": round(lat_max, 6),
                "nans_found": total_nans
            })
            
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_csv_path = os.path.join(results_dir, "dataset_inventory.csv")
    inventory_df.to_csv(inventory_csv_path, index=False)
    print(f"Dataset inventory saved to: {inventory_csv_path}")
    
    # Calculate sampling stats
    sampling_deltas = np.array(sampling_deltas)
    delta_1s_pct = float(np.mean(sampling_deltas == 1.0) * 100)
    delta_2s_pct = float(np.mean(sampling_deltas == 2.0) * 100)
    delta_gt2s_pct = float(np.mean(sampling_deltas > 2.0) * 100)
    
    eda_summary = {
        "total_trips": len(inventory_df),
        "train_trips": int(sum(inventory_df["partition"] == "train")),
        "student_test_trips": int(sum(inventory_df["partition"] == "student_test")),
        "total_gps_points": int(inventory_df["gps_points"].sum()),
        "avg_points_per_trip": round(float(inventory_df["gps_points"].mean()), 1),
        "min_points_per_trip": int(inventory_df["gps_points"].min()),
        "max_points_per_trip": int(inventory_df["gps_points"].max()),
        "total_route_arcs": int(inventory_df["route_arcs"].sum()),
        "avg_route_arcs_per_trip": round(float(inventory_df[inventory_df["route_arcs"] > 0]["route_arcs"].mean()), 1),
        "total_missing_values": missing_values_count,
        "non_monotonic_timestamp_trips": non_monotonic_timestamps,
        "sampling_delta_exact_1s_pct": round(delta_1s_pct, 2),
        "sampling_delta_2s_pct": round(delta_2s_pct, 2),
        "sampling_delta_gt2s_pct": round(delta_gt2s_pct, 2),
        "avg_trip_distance_km": round(float(inventory_df["distance_km"].mean()), 2),
        "avg_trip_speed_kmh": round(float(inventory_df["avg_speed_kmh"].mean()), 1)
    }
    
    summary_json_path = os.path.join(results_dir, "eda_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(eda_summary, f, indent=4)
    print(f"EDA summary JSON saved to: {summary_json_path}")
    
    # Generate Diagnostic Visualizations
    # 1. Sampling Interval Distribution
    plt.figure(figsize=(8, 5))
    dt_counts = pd.Series(sampling_deltas).value_counts(normalize=True).head(5) * 100
    sns.barplot(x=dt_counts.index.astype(str), y=dt_counts.values, palette="Blues_r")
    plt.title("GPS Sampling Interval Distribution (Nominal 1 Hz)", fontsize=14, pad=12, fontweight='bold')
    plt.xlabel("Sampling Interval Δt (seconds)", fontsize=12)
    plt.ylabel("Percentage of Steps (%)", fontsize=12)
    for i, v in enumerate(dt_counts.values):
        plt.text(i, v + 1.0, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    sampling_plot_path = os.path.join(results_dir, "eda_sampling_intervals.png")
    plt.savefig(sampling_plot_path, dpi=200)
    plt.close()
    
    # 2. Trip Point Count Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(data=inventory_df, x="gps_points", hue="partition", bins=25, kde=True, palette={"train": "#1f77b4", "student_test": "#ff7f0e"})
    plt.title("Distribution of GPS Observations per Trip", fontsize=14, pad=12, fontweight='bold')
    plt.xlabel("Number of GPS Points per Trip", fontsize=12)
    plt.ylabel("Trip Count", fontsize=12)
    plt.tight_layout()
    pts_plot_path = os.path.join(results_dir, "eda_trip_lengths.png")
    plt.savefig(pts_plot_path, dpi=200)
    plt.close()
    
    # 3. Network Size Distribution (Arcs vs Nodes)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=inventory_df, x="network_nodes", y="network_arcs", hue="partition", alpha=0.8, s=60)
    plt.title("Road-Network Size: Arcs vs Nodes per Trip", fontsize=14, pad=12, fontweight='bold')
    plt.xlabel("Number of Network Nodes", fontsize=12)
    plt.ylabel("Number of Road Arcs", fontsize=12)
    plt.tight_layout()
    net_plot_path = os.path.join(results_dir, "eda_network_sizes.png")
    plt.savefig(net_plot_path, dpi=200)
    plt.close()
    
    print("EDA Visualizations generated:")
    print(f"  - {sampling_plot_path}")
    print(f"  - {pts_plot_path}")
    print(f"  - {net_plot_path}")
    print("=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return eda_summary

if __name__ == "__main__":
    run_eda()
