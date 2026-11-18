"""
PathSnap: Feature Dataset Builder
Extracts candidates and features across training trips and student-testing trips.
Caches processed feature matrices to disk for fast model training and evaluation.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import load_trip
from features.candidate_extractor import CandidateExtractor

def build_dataset(
    data_dir: str = "data",
    results_dir: str = "results",
    max_train_trips: int = 30,
    stride: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(results_dir, exist_ok=True)
    train_cache = os.path.join(results_dir, "train_features.pkl.gz")
    test_cache = os.path.join(results_dir, "student_test_features.pkl.gz")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"Loading cached feature datasets from {results_dir}...")
        train_df = pd.read_pickle(train_cache)
        test_df = pd.read_pickle(test_cache)
        return train_df, test_df
        
    print("=" * 60)
    print("EXTRACTING FEATURES FOR TRAINING & STUDENT-TESTING")
    print("=" * 60)
    
    # 1. Training set
    train_dir = os.path.join(data_dir, "train")
    train_trips = sorted(os.listdir(train_dir))[:max_train_trips]
    train_dfs = []
    train_misses = []
    
    print(f"Extracting features from {len(train_trips)} training trips (stride={stride})...")
    t0 = time.time()
    for idx, tid in enumerate(train_trips):
        trip = load_trip(train_dir, tid, load_route=True)
        extractor = CandidateExtractor(trip, top_k=8)
        
        # Subsample points with stride for training efficiency
        records = []
        T = len(extractor.track_m)
        trip_miss = 0
        for t in range(0, T, stride):
            pt_records = extractor.extract_candidates_for_point(t)
            records.extend(pt_records)
            if extractor.point_true_arcs is not None:
                if not any(r["is_match"] == 1 for r in pt_records):
                    trip_miss += 1
        
        train_misses.append(trip_miss / max(len(range(0, T, stride)), 1))
        train_dfs.append(pd.DataFrame(records))
        if (idx + 1) % 5 == 0 or (idx + 1) == len(train_trips):
            print(f"  Processed {idx + 1}/{len(train_trips)} train trips...")
            
    train_df = pd.concat(train_dfs, ignore_index=True)
    avg_train_miss = np.mean(train_misses) * 100.0
    print(f"Train features shape: {train_df.shape}, Candidate Miss Rate: {avg_train_miss:.2f}%")
    
    # 2. Student Testing set (all 10 trips, full resolution, stride=1)
    test_dir = os.path.join(data_dir, "student_test")
    test_trips = sorted(os.listdir(test_dir))
    test_dfs = []
    test_misses = []
    
    print(f"\nExtracting features from {len(test_trips)} student-test trips (stride=1)...")
    for idx, tid in enumerate(test_trips):
        trip = load_trip(test_dir, tid, load_route=True)
        extractor = CandidateExtractor(trip, top_k=8)
        df_trip, miss_rate = extractor.extract_all()
        test_dfs.append(df_trip)
        test_misses.append(miss_rate)
        print(f"  Trip {tid}: {len(trip.track)} points, {len(df_trip)} candidates, Miss Rate: {miss_rate:.2f}%")
        
    test_df = pd.concat(test_dfs, ignore_index=True)
    avg_test_miss = np.mean(test_misses)
    print(f"Student test features shape: {test_df.shape}, Candidate Miss Rate: {avg_test_miss:.2f}%")
    
    # Save cache
    train_df.to_pickle(train_cache, compression='gzip')
    test_df.to_pickle(test_cache, compression='gzip')
    print(f"Cached train features to {train_cache}")
    print(f"Cached test features to {test_cache}")
    print(f"Total feature extraction time: {time.time() - t0:.2f}s")
    print("=" * 60)
    
    return train_df, test_df

if __name__ == "__main__":
    build_dataset()
