# PATHSNAP: Distributed Map-Matching & Route Reconstruction System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![PES University](https://img.shields.io/badge/PES%20University-UE24CS352A-orange.svg)](https://pes.edu/)

PathSnap is an end-to-end distributed machine learning system developed for **UE24CS352A: Machine Learning** at PES University. The system reconstructs vehicle trajectories by matching noisy, multipath-degraded GPS measurements to an underlying directed road network graph.

---

## Key Highlights & Results

* **Multi-Family ML Comparison:** Trained and cross-validated 5 distinct model families (**Logistic Regression, KNN, Random Forest, LightGBM, MLP**) using 5-fold GroupKFold by trip.
* **Bayesian Hyperparameter Optimization:** Optimized LightGBM using Optuna (TPE) across 20 trials, reducing log loss to **0.1477**.
* **HMM & Viterbi Sequential Reconstruction:** Combines spatial Gaussian likelihood, ML emission confidence, and topological graph reachability to achieve **99.59% Route Continuity** and **80.72% average Pointwise Accuracy** on the official 10 student-test trips.
* **Two-Laptop Distributed Architecture:** Real-time inference via FastAPI on Laptop 1 with live CMD logging, connecting to an evaluator dashboard on Laptop 2.

---

## Repository Structure

```text
PathSnap/
├── data/
│   ├── train/                 # 80 Training trips (00000001 - 00000100)
│   └── student_test/          # 10 Student Testing trips (00000000, 00000007, ...)
├── data_processing/
│   ├── __init__.py
│   ├── loader.py              # Loads .track, .nodes, .arcs, .route with metric scaling
│   └── eda.py                 # Exploratory data analysis, inventory & diagnostic plots
├── features/
│   ├── __init__.py
│   ├── geo_utils.py           # Haversine, bearing, orthogonal projection & graph tools
│   ├── candidate_extractor.py # 3-Point KDTree spatial index & 8-feature extractor
│   └── build_features.py      # Feature matrix caching script
├── models/
│   ├── __init__.py
│   └── train_all.py           # Trains 5 models, evaluates GroupKFold CV, Top-1 & Continuity
├── optimization/
│   ├── __init__.py
│   └── bayesian_opt.py        # Optuna Bayesian hyperparameter search (20 trials)
├── hmm/
│   ├── __init__.py
│   ├── viterbi.py             # Log-space Viterbi algorithm & local graph reachability
│   └── evaluation.py          # Benchmark comparing Classifier vs HMM on test set
├── visualizations/
│   ├── __init__.py
│   ├── folium_map.py          # Multi-layer interactive Folium HTML map generator
│   └── plot_metrics.py        # Publication-ready diagnostic charts & confusion matrix
├── api/
│   ├── __init__.py
│   └── server.py              # FastAPI server for Laptop 1 (evaluator endpoints & logs)
├── ui/
│   ├── __init__.py
│   ├── templates/
│   │   └── index.html         # Responsive evaluator dashboard for Laptop 2
│   └── client.py              # UI client runner (port 3000)
├── saved_models/              # Pickled models, scalers, and best tuned weights
├── results/                   # Generated evaluation CSVs, charts, and Folium HTML maps
├── report/
│   └── PathSnap_Technical_Report.md  # Consolidated Technical Report
├── requirements.txt
└── README.md
```

---

##  Installation & Setup

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone <your-github-repo-url>
cd "ML Hackathon"

# Install all project dependencies
pip install -r requirements.txt
```

---

## Pipeline Execution (Step-by-Step)

To reproduce the experiments from scratch:

```bash
# 1. Run Exploratory Data Analysis (EDA)
python data_processing/eda.py

# 2. Extract and Cache Features
python features/build_features.py

# 3. Train and Compare All 5 Models
python models/train_all.py

# 4. Generate Section 3.3 Diagnostic Plots
python visualizations/plot_metrics.py

# 5. Run Bayesian Hyperparameter Optimization (Optuna)
python optimization/bayesian_opt.py

# 6. Run HMM Route Reconstruction on Student-Test Trips
python hmm/evaluation.py
```

---

## Two-Laptop Live Demonstration Guide

The project is structured according to the **Two-Laptop Deployment Architecture**:

```
┌──────────────────────────────────────────────┐              HTTP REST              ┌──────────────────────────────────────────────┐
│                   LAPTOP 2                   │        POST /api/evaluate/{id}      │                   LAPTOP 1                   │
│             (Evaluator Dashboard)            ├────────────────────────────────────►│           (FastAPI Inference Server)         │
│                                              │                                     │                                              │
│ • Runs client UI at port 3000 (or web)       │    Evaluation Metrics & Folium Map  │ • Hosts LightGBM tuned model + HMM decoder   │
│ • Lists the 10 evaluation trips              │◄────────────────────────────────────┤ • Holds server-side reference routes (.route)│
│ • Dispatches trip ID to Laptop 1 IP          │                                     │ • Evaluates ground-truth accuracy on-the-fly │
│ • Embeds interactive Leaflet/Folium map      │                                     │ • Emits detailed live inference logs to CMD  │
└──────────────────────────────────────────────┘                                     └──────────────────────────────────────────────┘
```

### Laptop 1 (Backend API Host):
1. Find your local IP address:
   ```cmd
   ipconfig
   # Look for IPv4 Address, e.g., 192.168.1.15
   ```
2. Start the FastAPI backend server:
   ```bash
   python api/server.py
   ```
3. The server starts listening on `0.0.0.0:8000` with live terminal logging.

### Laptop 2 (Evaluator User Interface):
1. Connect Laptop 2 to the same WiFi network as Laptop 1.
2. Open a web browser and navigate directly to:
   ```
   http://<LAPTOP_1_IP>:8000/ui
   # (e.g., http://192.168.1.15:8000/ui)
   ```
   *Alternatively, run `python ui/client.py` on Laptop 2 and open `http://localhost:3000`.*
3. Select any one of the 10 reserved evaluation trips from the dropdown.
4. Click **Run Evaluation Pipeline**.
5. Within ~500 ms, the UI renders:
   * **Observation-level Pointwise Top-1 Accuracy**
   * **Route Continuity**
   * **Sequence Recovery F1-Score**
   * **Interactive Folium Map** (Raw GPS blue markers, Ground-truth green route, HMM predicted red route).
6. The evaluator can verify the authentic incoming request and live execution on the Laptop 1 CMD terminal.

---

## Benchmark Summary

### 1. Model Comparison (Table 3.1)
| Model | Algorithm Family | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 0.8157 | 65.85% | 88.61% | 0.8183 | 0.1692 | 0.000 ms |
| **KNN (Distance-Weighted)** | Instance-Based | 0.8069 | 64.55% | 87.02% | 0.8154 | 0.5302 | 0.029 ms |
| **Random Forest** | Ensemble | 0.8291 | 66.86% | 89.51% | 0.8328 | 0.1491 | 0.002 ms |
| **LightGBM** | Boosting | **0.8294** | 66.47% | 88.98% | **0.8330** | 0.1585 | **0.001 ms** |
| **MLP Classifier** | Neural Networks | 0.8247 | 65.59% | 88.28% | 0.8282 | 0.1518 | 0.001 ms |

### 2. Bayesian Tuning (Untuned vs Tuned LightGBM)
* **CV Macro F1:** $0.8294 \rightarrow \mathbf{0.8337}$
* **Student-Test Top-1 Accuracy:** $66.47\% \rightarrow \mathbf{66.84\%}$
* **Log Loss:** $0.1585 \rightarrow \mathbf{0.1477}$ (-6.8% reduction)

### 3. HMM Sequential Reconstruction
* **Average Pointwise Accuracy:** $75.91\% \rightarrow \mathbf{80.72\%}$ (+4.81% gain)
* **Average Route Continuity:** $95.74\% \rightarrow \mathbf{99.59\%}$ (+3.85% gain)
* **Average Sequence Route F1:** **0.8670** (Trip 00000007 reached **0.9719**)

---

## Academic Integrity Declaration
All components of this project were developed independently by the team without unauthorized collaboration or data leakage. Evaluator testing data was strictly held out and untouched. Reference routes remain server-side and are never hardcoded in the client interface.
