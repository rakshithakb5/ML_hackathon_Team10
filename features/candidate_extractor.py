"""
PathSnap: Spatial Candidate Generator & Feature Extractor
Uses KDTree spatial indexing on arc endpoints and midpoints to retrieve top-K candidate segments per GPS observation.
Extracts geometric, kinematic, and topological features for classification and HMM emission.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import TripDataLoader, load_trip
from features.geo_utils import (
    project_point_to_segments,
    compute_bearing,
    angular_difference,
    haversine_distance
)

class CandidateExtractor:
    """Extracts candidate segments and geometric features for a single trip."""
    
    def __init__(self, trip: TripDataLoader, top_k: int = 8):
        self.trip = trip
        self.top_k = top_k
        
        # Metric coordinates
        self.nodes_m = self.trip.to_meters(self.trip.nodes)
        self.from_m = self.nodes_m[self.trip.arcs[:, 0]]
        self.to_m = self.nodes_m[self.trip.arcs[:, 1]]
        self.mid_m = (self.from_m + self.to_m) / 2.0
        
        # Precompute arc segment bearings and lengths
        seg_vec = self.to_m - self.from_m
        self.arc_lengths = np.sqrt(np.maximum(np.sum(seg_vec**2, axis=1), 1e-12))
        self.arc_bearings = (np.degrees(np.arctan2(seg_vec[:, 0], seg_vec[:, 1])) + 360.0) % 360.0
        
        # Multi-point spatial index: start, mid, end of each arc
        points_to_index = np.vstack([self.from_m, self.mid_m, self.to_m])
        self.arc_map = np.tile(np.arange(len(self.from_m)), 3)
        self.kdtree = KDTree(points_to_index)
        
        # Precompute GPS kinematics
        self.track_m = self.trip.to_meters(self.trip.track[:, :2])
        self.timestamps = self.trip.track[:, 2]
        self._compute_gps_kinematics()
        
        # Ground truth route mapping if route is present
        self.point_true_arcs = None
        if self.trip.route is not None:
            self._map_ground_truth_route()

    def _compute_gps_kinematics(self):
        """Computes GPS speeds and headings."""
        T = len(self.track_m)
        self.gps_speeds = np.zeros(T, dtype=np.float32)
        self.gps_headings = np.zeros(T, dtype=np.float32)
        
        if T > 1:
            diffs = np.diff(self.track_m, axis=0)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dts = np.maximum(np.diff(self.timestamps), 0.1)
            speeds = dists / dts
            headings = (np.degrees(np.arctan2(diffs[:, 0], diffs[:, 1])) + 360.0) % 360.0
            
            self.gps_speeds[:-1] = speeds
            self.gps_speeds[-1] = speeds[-1] if len(speeds) > 0 else 0.0
            
            self.gps_headings[:-1] = headings
            self.gps_headings[-1] = headings[-1] if len(headings) > 0 else 0.0

    def _map_ground_truth_route(self):
        """Maps each GPS point to its active ground truth route arc using monotonic forward matching."""
        route = self.trip.route
        R = len(route)
        T = len(self.track_m)
        self.point_true_arcs = np.zeros(T, dtype=np.int64)
        
        route_from = self.from_m[route]
        route_to = self.to_m[route]
        
        curr_r_idx = 0
        window_size = 150
        
        for t in range(T):
            p = self.track_m[t]
            start_idx = max(0, curr_r_idx - 10)
            end_idx = min(R, curr_r_idx + window_size)
            
            sub_from = route_from[start_idx:end_idx]
            sub_to = route_to[start_idx:end_idx]
            
            dists, _, _, _, _ = project_point_to_segments(p, sub_from, sub_to)
            best_rel_idx = int(np.argmin(dists))
            curr_r_idx = start_idx + best_rel_idx
            self.point_true_arcs[t] = route[curr_r_idx]

    def extract_candidates_for_point(self, point_idx: int) -> List[Dict[str, Any]]:
        """Extracts candidate segments and features for a single GPS observation."""
        p_xy = self.track_m[point_idx]
        speed = float(self.gps_speeds[point_idx])
        heading = float(self.gps_headings[point_idx])
        true_arc = int(self.point_true_arcs[point_idx]) if self.point_true_arcs is not None else -1
        
        # Query KDTree for nearest index points
        query_k = min(self.top_k * 4, len(self.arc_map))
        _, query_indices = self.kdtree.query(p_xy, k=query_k)
        if np.isscalar(query_indices):
            query_indices = np.array([query_indices])
            
        # Get unique arc candidates
        unique_cands = np.unique(self.arc_map[query_indices])
        
        c_from = self.from_m[unique_cands]
        c_to = self.to_m[unique_cands]
        c_bearings = self.arc_bearings[unique_cands]
        
        dists, proj_t, d_starts, d_ends, seg_lens = project_point_to_segments(p_xy, c_from, c_to)
        head_diffs = angular_difference(c_bearings, heading)
        
        # Sort candidates by perpendicular distance and take top_k
        sort_order = np.argsort(dists)[:self.top_k]
        
        candidates = []
        for rank, idx in enumerate(sort_order):
            arc_id = int(unique_cands[idx])
            is_match = 1 if (true_arc != -1 and arc_id == true_arc) else 0
            
            candidates.append({
                "trip_id": self.trip.trip_id,
                "point_idx": point_idx,
                "candidate_rank": rank,
                "arc_id": arc_id,
                "from_node": int(self.trip.arcs[arc_id, 0]),
                "to_node": int(self.trip.arcs[arc_id, 1]),
                "dist_to_segment": float(dists[idx]),
                "heading_diff": float(head_diffs[idx]),
                "dist_to_start": float(d_starts[idx]),
                "dist_to_end": float(d_ends[idx]),
                "proj_ratio": float(proj_t[idx]),
                "segment_length": float(seg_lens[idx]),
                "gps_speed": speed,
                "is_match": is_match
            })
            
        return candidates

    def extract_all(self) -> Tuple[pd.DataFrame, float]:
        """Extracts candidates across all points in the trajectory."""
        all_records = []
        misses = 0
        T = len(self.track_m)
        
        for t in range(T):
            records = self.extract_candidates_for_point(t)
            all_records.extend(records)
            if self.point_true_arcs is not None:
                has_match = any(r["is_match"] == 1 for r in records)
                if not has_match:
                    misses += 1
                    
        df = pd.DataFrame(all_records)
        miss_rate = (misses / max(T, 1)) * 100.0 if self.point_true_arcs is not None else 0.0
        return df, miss_rate
