"""
PathSnap: Distributed Backend API (Laptop 1)
FastAPI service hosting:
1. Trained LightGBM classifier & feature extraction pipeline
2. HMM & Viterbi sequential route reconstructor
3. Server-side evaluation & ground-truth validation (no leakage to UI)
4. Interactive Folium map generation and streaming
5. Detailed terminal logging for evaluator demonstration
"""

import os
import sys
import time
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import load_trip, TripDataLoader
from features.candidate_extractor import CandidateExtractor
from hmm.viterbi import HMMMapMatcher
from visualizations.folium_map import generate_trip_folium_map

# Configure Terminal Logging for Evaluator Inspection
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Backend service hosting ML models, HMM sequential decoding, and server-side evaluation.",
    version="1.0.0"
)

# Enable CORS for multi-laptop communication across local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached model and directories
MODEL_PATH = "saved_models/lightgbm_tuned.joblib"
DATA_DIR = "data"
EVAL_PARTITIONS = ["student_test", "evaluator_test", "train"]
CLF_MODEL = None

@app.on_event("startup")
def startup_event():
    global CLF_MODEL
    logger.info("=" * 65)
    logger.info("INITIALIZING PATHSNAP BACKEND SERVER (LAPTOP 1)")
    logger.info("=" * 65)
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading trained model artifact from: {MODEL_PATH}")
        CLF_MODEL = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully into memory.")
    else:
        logger.warning(f"Model file {MODEL_PATH} not found. Running with baseline fallback.")
    logger.info("Ready to accept incoming evaluation requests.")
    logger.info("=" * 65)

def find_trip_partition(trip_id: str) -> str:
    """Locates which directory holds the requested trip_id."""
    for p in EVAL_PARTITIONS:
        path = os.path.join(DATA_DIR, p, trip_id)
        if os.path.exists(path):
            return os.path.join(DATA_DIR, p)
    raise FileNotFoundError(f"Trip ID '{trip_id}' not found in any data partition.")

@app.get("/")
def root():
    return {
        "system": "PathSnap: Distributed Map-Matching & Route Reconstruction",
        "role": "Laptop 1 (Model and API Host)",
        "status": "online",
        "model_loaded": CLF_MODEL is not None,
        "ui_url": "/ui",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ui", response_class=HTMLResponse)
def serve_evaluator_ui():
    """Serves the evaluator frontend UI directly from Laptop 1 for convenience."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "templates", "index.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/trips")
def list_evaluation_trips():
    """Lists all available evaluation trips (from student_test partition)."""
    target_dir = os.path.join(DATA_DIR, "student_test")
    if not os.path.exists(target_dir):
        # Fallback to any partition available
        for p in EVAL_PARTITIONS:
            candidate = os.path.join(DATA_DIR, p)
            if os.path.exists(candidate):
                target_dir = candidate
                break
                
    if not os.path.exists(target_dir):
        return {"trips": []}
        
    trips = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    logger.info(f"Served list of {len(trips)} evaluation trips to client.")
    return {"partition": os.path.basename(target_dir), "trips": trips}

@app.post("/api/evaluate/{trip_id}")
def evaluate_trip(trip_id: str):
    """
    Executes the full end-to-end map-matching pipeline on the selected trip:
    1. Loads raw .track, .nodes, .arcs
    2. Candidate generation & feature extraction
    3. Tuned LightGBM scoring
    4. HMM Viterbi global route reconstruction
    5. Calculates prediction metrics against server-side ground truth (.route)
    6. Generates interactive Folium map
    """
    t_start = time.time()
    logger.info(f">>> [INCOMING REQUEST] Evaluator selected Trip: {trip_id}")
    
    try:
        partition_path = find_trip_partition(trip_id)
    except FileNotFoundError:
        logger.error(f"Trip {trip_id} not found in database.")
        raise HTTPException(status_code=404, detail=f"Trip ID {trip_id} not found.")
        
    # 1. Load data
    logger.info(f"Loading data for trip {trip_id} from {partition_path}...")
    trip = load_trip(partition_path, trip_id, load_route=True)
    n_pts = len(trip.track)
    logger.info(f"Trip {trip_id}: {n_pts} GPS observations, {len(trip.arcs)} road arcs, {len(trip.nodes)} nodes.")
    
    # 2. Candidate Extraction
    t_extract_start = time.time()
    extractor = CandidateExtractor(trip, top_k=8)
    extract_time = time.time() - t_extract_start
    logger.info(f"Candidate extraction completed in {extract_time:.2f}s.")
    
    # 3. Model Inference & HMM Viterbi Sequential Decoding
    t_hmm_start = time.time()
    matcher = HMMMapMatcher(trip, CLF_MODEL)
    result = matcher.run_reconstruction(extractor)
    hmm_time = time.time() - t_hmm_start
    logger.info(f"HMM Viterbi decoding completed in {hmm_time:.2f}s.")
    
    # 4. Server-Side Ground Truth Evaluation
    true_route_set = set(trip.route) if trip.route is not None else set()
    pred_route_set = set(result["reconstructed_route"])
    
    if len(true_route_set) > 0:
        intersection = len(pred_route_set.intersection(true_route_set))
        precision = (intersection / max(len(pred_route_set), 1)) * 100.0
        recall = (intersection / max(len(true_route_set), 1)) * 100.0
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6) / 100.0
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
        
    total_time_ms = round((time.time() - t_start) * 1000.0, 1)
    
    # Log evaluation results to terminal
    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS FOR TRIP: {trip_id}")
    logger.info(f"  Observations:             {n_pts} GPS points")
    logger.info(f"  Classifier Pointwise Acc: {result['observation_acc_clf']:.2f}%")
    logger.info(f"  HMM Pointwise Accuracy:   {result['observation_acc_hmm']:.2f}%")
    logger.info(f"  Classifier Continuity:    {result['continuity_clf']:.2f}%")
    logger.info(f"  HMM Route Continuity:     {result['continuity_hmm']:.2f}%")
    logger.info(f"  Reconstructed Arcs Count: {len(result['reconstructed_route'])}")
    logger.info(f"  Reference Arcs Count:     {len(true_route_set)}")
    logger.info(f"  Route Sequence F1-Score:  {f1:.4f}")
    logger.info(f"  Total Pipeline Latency:   {total_time_ms} ms")
    logger.info("=" * 60)
    
    # 5. Generate interactive Folium Map
    map_dir = "results"
    os.makedirs(map_dir, exist_ok=True)
    map_filename = f"map_trip_{trip_id}.html"
    map_filepath = os.path.join(map_dir, map_filename)
    generate_trip_folium_map(
        trip=trip,
        predicted_arc_ids=result["reconstructed_route"],
        output_html_path=map_filepath
    )
    
    response_payload = {
        "trip_id": trip_id,
        "gps_points_count": n_pts,
        "observation_acc_clf": result["observation_acc_clf"],
        "observation_acc_hmm": result["observation_acc_hmm"],
        "continuity_clf": result["continuity_clf"],
        "continuity_hmm": result["continuity_hmm"],
        "reconstructed_arcs_count": len(result["reconstructed_route"]),
        "reference_arcs_count": len(true_route_set),
        "route_precision_pct": round(precision, 2),
        "route_recall_pct": round(recall, 2),
        "route_f1_score": round(f1, 4),
        "total_latency_ms": total_time_ms,
        "map_url": f"/api/map/{trip_id}",
        "status": "success"
    }
    logger.info(f"<<< [RESPONSE DISPATCHED] Status: 200 OK to Evaluator UI.")
    return response_payload

@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_trip_map(trip_id: str):
    """Serves the generated interactive Folium map for the trip."""
    map_filepath = os.path.join("results", f"map_trip_{trip_id}.html")
    if not os.path.exists(map_filepath):
        # Generate on the fly
        partition_path = find_trip_partition(trip_id)
        trip = load_trip(partition_path, trip_id, load_route=True)
        extractor = CandidateExtractor(trip, top_k=8)
        matcher = HMMMapMatcher(trip, CLF_MODEL)
        res = matcher.run_reconstruction(extractor)
        generate_trip_folium_map(trip, res["reconstructed_route"], map_filepath)
        
    with open(map_filepath, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so Laptop 2 on local network can connect
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
