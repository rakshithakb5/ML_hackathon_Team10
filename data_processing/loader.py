"""
PathSnap: Data Loading and Validation Utilities
Handles loading of .track, .nodes, .arcs, and .route files.
Converts angular coordinates to local metric coordinates (meters).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class TripDataLoader:
    """Loads and caches road-network and GPS trajectory data for a single trip."""
    
    def __init__(self, trip_dir: str, trip_id: str):
        self.trip_dir = trip_dir
        self.trip_id = trip_id
        
        self.track_path = os.path.join(trip_dir, f"{trip_id}.track")
        self.nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
        self.arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
        self.route_path = os.path.join(trip_dir, f"{trip_id}.route")
        
        self.track: Optional[np.ndarray] = None  # (T, 3): lon, lat, timestamp
        self.nodes: Optional[np.ndarray] = None  # (N, 2): lon, lat
        self.arcs: Optional[np.ndarray] = None   # (M, 2): from_node, to_node
        self.route: Optional[np.ndarray] = None  # (R,): arc_ids
        
        self.center_lat = 49.85
        self.scale_x = 111139.0 * np.cos(np.radians(self.center_lat))
        self.scale_y = 111139.0

    def load(self, load_route: bool = True) -> "TripDataLoader":
        """Loads all trip files."""
        if not os.path.exists(self.track_path):
            raise FileNotFoundError(f"Track file not found: {self.track_path}")
        if not os.path.exists(self.nodes_path):
            raise FileNotFoundError(f"Nodes file not found: {self.nodes_path}")
        if not os.path.exists(self.arcs_path):
            raise FileNotFoundError(f"Arcs file not found: {self.arcs_path}")
            
        self.track = np.loadtxt(self.track_path, dtype=np.float64)
        if self.track.ndim == 1:
            self.track = self.track.reshape(1, -1)
            
        self.nodes = np.loadtxt(self.nodes_path, dtype=np.float64)
        if self.nodes.ndim == 1:
            self.nodes = self.nodes.reshape(1, -1)
            
        self.arcs = np.loadtxt(self.arcs_path, dtype=np.int64)
        if self.arcs.ndim == 1:
            self.arcs = self.arcs.reshape(1, -1)
            
        if load_route and os.path.exists(self.route_path):
            try:
                self.route = np.loadtxt(self.route_path, dtype=np.int64)
                if self.route.ndim == 0:
                    self.route = np.array([int(self.route)], dtype=np.int64)
            except Exception:
                self.route = None
        else:
            self.route = None
            
        # Update center latitude from actual nodes for accurate projection
        if len(self.nodes) > 0:
            self.center_lat = float(np.mean(self.nodes[:, 1]))
            self.scale_x = 111139.0 * np.cos(np.radians(self.center_lat))
            self.scale_y = 111139.0
            
        return self

    def to_meters(self, lon_lat: np.ndarray) -> np.ndarray:
        """Projects [lon, lat] degrees into local Cartesian coordinates (meters)."""
        x = lon_lat[:, 0] * self.scale_x
        y = lon_lat[:, 1] * self.scale_y
        return np.column_stack([x, y])

    def to_degrees(self, xy_meters: np.ndarray) -> np.ndarray:
        """Converts local metric coordinates back to [lon, lat] degrees."""
        lon = xy_meters[:, 0] / self.scale_x
        lat = xy_meters[:, 1] / self.scale_y
        return np.column_stack([lon, lat])

    def get_arc_endpoints_meters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns metric coordinates of start (from) and end (to) nodes for all arcs."""
        nodes_m = self.to_meters(self.nodes)
        from_pts = nodes_m[self.arcs[:, 0]]
        to_pts = nodes_m[self.arcs[:, 1]]
        return from_pts, to_pts

    def get_track_meters(self) -> np.ndarray:
        """Returns metric coordinates of the GPS trajectory points."""
        return self.to_meters(self.track[:, :2])


def load_trip(partition_dir: str, trip_id: str, load_route: bool = True) -> TripDataLoader:
    """Helper function to load a trip by partition directory and trip ID."""
    trip_dir = os.path.join(partition_dir, trip_id)
    loader = TripDataLoader(trip_dir, trip_id)
    return loader.load(load_route=load_route)
