"""
PathSnap: Geospatial, Geometric, and Graph Utilities
Implements Haversine distance, bearings, angular differences,
orthogonal point-to-segment projections, and NetworkX graph builders.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any, Optional

def haversine_distance(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance between two sets of points on Earth in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def compute_bearing(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Computes initial compass bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)
    
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360.0) % 360.0

def angular_difference(deg1: np.ndarray, deg2: np.ndarray) -> np.ndarray:
    """Computes shortest angular difference between two bearings in degrees [0, 180]."""
    diff = np.abs(deg1 - deg2) % 360.0
    return np.minimum(diff, 360.0 - diff)

def project_point_to_segments(
    p_xy: np.ndarray,          # (2,) point in meters
    a_pts: np.ndarray,         # (K, 2) segment start points in meters
    b_pts: np.ndarray          # (K, 2) segment end points in meters
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects a single 2D metric point onto K line segments.
    Returns:
        dists: (K,) perpendicular distance in meters
        proj_ratios: (K,) projection scalar t in [0, 1]
        dist_to_starts: (K,) distance to segment start node
        dist_to_ends: (K,) distance to segment end node
        seg_lengths: (K,) segment lengths in meters
    """
    v = b_pts - a_pts  # (K, 2) segment direction vectors
    seg_lengths_sq = np.sum(v**2, axis=1)  # (K,)
    seg_lengths = np.sqrt(np.maximum(seg_lengths_sq, 1e-12))
    
    u = p_xy - a_pts  # (K, 2)
    dot = np.sum(u * v, axis=1)  # (K,)
    
    # Projection scalar clamped to [0, 1]
    t = np.clip(dot / np.maximum(seg_lengths_sq, 1e-12), 0.0, 1.0)
    
    closest_pts = a_pts + t[:, None] * v  # (K, 2)
    dists = np.sqrt(np.sum((p_xy - closest_pts)**2, axis=1))
    
    dist_to_starts = np.sqrt(np.sum(u**2, axis=1))
    dist_to_ends = np.sqrt(np.sum((p_xy - b_pts)**2, axis=1))
    
    return dists, t, dist_to_starts, dist_to_ends, seg_lengths

def build_road_graph(arcs: np.ndarray, arc_lengths: np.ndarray) -> nx.DiGraph:
    """Builds a directed NetworkX graph representing the road network with arc weights."""
    G = nx.DiGraph()
    for arc_id, (u, v) in enumerate(arcs):
        length = float(arc_lengths[arc_id]) if arc_lengths is not None else 1.0
        G.add_edge(int(u), int(v), arc_id=arc_id, weight=length)
    return G
