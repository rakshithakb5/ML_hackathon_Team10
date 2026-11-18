"""
PathSnap: Folium Geographical Visualizer
Generates interactive, multi-layered Leaflet/Folium HTML maps:
- Original noisy GPS observations (Blue markers)
- Road network geometry (Thin gray lines)
- Ground-truth reference route (Solid Green polyline)
- Predicted map-matched route (Dashed Red polyline)
Includes layer controls, legend, popups, and auto-zoom to trajectory extent.
"""

import os
import sys
import folium
from folium import plugins
import numpy as np
from typing import List, Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import TripDataLoader, load_trip

def generate_trip_folium_map(
    trip: TripDataLoader,
    predicted_arc_ids: Optional[List[int]] = None,
    output_html_path: str = "results/trip_map.html",
    include_road_network: bool = True
) -> str:
    """
    Creates an interactive Folium map for a single trip.
    Layers:
    1. Base Map (CartoDB Positron & OpenStreetMap)
    2. Road Network (subsampled or surrounding arcs in light gray)
    3. Ground Truth Reference Route (Green)
    4. Predicted Route (Red)
    5. Raw GPS Observations (Blue circle markers)
    """
    os.makedirs(os.path.dirname(output_html_path) or ".", exist_ok=True)
    
    # Calculate center coordinates [lat, lon]
    lons = trip.track[:, 0]
    lats = trip.track[:, 1]
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    
    # Initialize Folium Map with OpenStreetMap (no API key required)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    # FeatureGroups for Layer Control
    fg_road_net = folium.FeatureGroup(name="Road Network (Grey)", show=False)
    fg_truth_route = folium.FeatureGroup(name="Reference Ground Truth (Green)", show=True)
    fg_pred_route = folium.FeatureGroup(name="Predicted Map-Matched Route (Red)", show=True)
    fg_gps_pts = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    
    # 1. Road Network Layer (arcs within bounding box of track)
    if include_road_network:
        bbox_pad = 0.01
        lon_min, lon_max = np.min(lons) - bbox_pad, np.max(lons) + bbox_pad
        lat_min, lat_max = np.min(lats) - bbox_pad, np.max(lats) + bbox_pad
        
        # Subsample arcs in bounding box to keep map responsive
        node_coords = trip.nodes
        for arc_id in range(0, len(trip.arcs), 2):
            u, v = trip.arcs[arc_id]
            p1_lon, p1_lat = node_coords[u]
            p2_lon, p2_lat = node_coords[v]
            if (lon_min <= p1_lon <= lon_max and lat_min <= p1_lat <= lat_max):
                folium.PolyLine(
                    locations=[[p1_lat, p1_lon], [p2_lat, p2_lon]],
                    color="#aaaaaa",
                    weight=1.5,
                    opacity=0.5
                ).add_to(fg_road_net)
                
    # 2. Ground Truth Route (Green)
    if trip.route is not None and len(trip.route) > 0:
        route_lines = []
        for arc_id in trip.route:
            u, v = trip.arcs[arc_id]
            p1 = [trip.nodes[u, 1], trip.nodes[u, 0]]
            p2 = [trip.nodes[v, 1], trip.nodes[v, 0]]
            route_lines.append([p1, p2])
            
        for seg in route_lines:
            folium.PolyLine(
                locations=seg,
                color="#28a745",
                weight=4.5,
                opacity=0.85,
                tooltip="Ground Truth Route"
            ).add_to(fg_truth_route)
            
    # 3. Predicted Route (Red)
    if predicted_arc_ids is not None and len(predicted_arc_ids) > 0:
        pred_lines = []
        for arc_id in predicted_arc_ids:
            if arc_id < len(trip.arcs):
                u, v = trip.arcs[arc_id]
                p1 = [trip.nodes[u, 1], trip.nodes[u, 0]]
                p2 = [trip.nodes[v, 1], trip.nodes[v, 0]]
                pred_lines.append([p1, p2])
                
        for seg in pred_lines:
            folium.PolyLine(
                locations=seg,
                color="#e63946",
                weight=3.5,
                opacity=0.9,
                dash_array="6, 4",
                tooltip="Predicted Route (HMM)"
            ).add_to(fg_pred_route)
            
    # 4. Raw GPS Observations (Blue Dots)
    # Subsample GPS markers if trajectory is very long for snappy UI
    step_stride = max(1, len(trip.track) // 300)
    for i in range(0, len(trip.track), step_stride):
        lon, lat, ts = trip.track[i]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#1d3557",
            fill=True,
            fill_color="#457b9d",
            fill_opacity=0.8,
            popup=f"GPS #{i}<br>Time: {ts:.1f}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        ).add_to(fg_gps_pts)
        
    # Add FeatureGroups to Map
    fg_road_net.add_to(m)
    fg_truth_route.add_to(m)
    fg_pred_route.add_to(m)
    fg_gps_pts.add_to(m)
    
    # Layer Control & Fullscreen Plugin
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    
    # Custom HTML Legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 25px; left: 25px; width: 220px; height: 130px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; font-family: Arial, sans-serif; padding: 10px;
                border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b style="font-size: 13px;">PathSnap Map Legend</b><br>
        <span style="color:#1d3557; font-size:16px;">●</span> Raw GPS Points<br>
        <span style="color:#28a745; font-size:18px; font-weight:bold;">—</span> Ground Truth Route<br>
        <span style="color:#e63946; font-size:18px; font-weight:bold;">- -</span> Predicted Route (HMM)<br>
        <span style="color:#aaaaaa; font-size:18px; font-weight:bold;">—</span> Road Network Arcs<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save to HTML
    m.save(output_html_path)
    print(f"Interactive Folium map saved to: {output_html_path}")
    return output_html_path
