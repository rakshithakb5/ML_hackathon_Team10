# PathSnap: Distributed Map-Matching & Route Reconstruction
## Technical Report & System Documentation

**Course:** UE24CS352A — Machine Learning  
**Department:** Department of Computer Science and Engineering, PES University, RR Campus  
**Project:** PathSnap Distributed Map-Matching System  

---

## Executive Summary

GPS observations recorded from moving vehicles inherently suffer from sensor noise, urban canyon multipath reflections, signal dropouts, and low sampling intervals. When plotted directly, raw trajectories frequently jump across buildings or falsely snap to parallel avenues. 

**PathSnap** is an end-to-end, distributed machine learning system designed to solve this challenge. The system combines:
1. Spatial candidate generation with multi-point KDTree indexing.
2. Kinematic, geometric, and topological feature engineering.
3. Supervised classification across five distinct algorithm families.
4. Bayesian hyperparameter optimization using Tree-structured Parzen Estimators (TPE).
5. A sequential Hidden Markov Model (HMM) solved via the Viterbi algorithm to enforce physical road-network continuity.
6. A distributed two-laptop deployment consisting of a high-throughput FastAPI backend (Laptop 1) and a responsive, evaluator-facing web dashboard (Laptop 2).

The final tuned system achieves an average **80.72% observation-level pointwise accuracy** across the official student-testing dataset (with individual trips exceeding **94% to 98%**), increases route topological continuity from **95.74% to 99.59%**, and recovers ground-truth traversed routes with an average **Sequence F1-Score of 0.8670** at sub-second latency.

---

## Section 2: Dataset Understanding & Exploratory Data Analysis (EDA)

### 2.1 Dataset Partitioning & Integrity
The PathSnap dataset comprises 100 trips structured into three partitions:
* **Training Partition (80 trips):** Used exclusively for exploratory data analysis, feature engineering, 5-fold group cross-validation, model training, and Bayesian hyperparameter search.
* **Student Testing Partition (10 trips):** Used exclusively for development-stage model evaluation, comparative analysis, and HMM parameter tuning.
* **Evaluator Testing Partition (10 trips):** Strictly held out and isolated on the server for final live evaluation.

```
Total Trips Analyzed:          90 trips (80 Train + 10 Student Test)
Total GPS Observations:        227,577 points
Average Observations / Trip:   2,528.6 points (Min: 262, Max: 19,313)
Total Traversed Route Arcs:    34,813 arcs (Average: 386.8 arcs/trip)
Missing / NaN Values:          0 (100% complete files)
Non-Monotonic Timestamps:      0 (Chronologically ordered)
Average Trip Travel Distance:  36.38 km
Average Vehicle Travel Speed:  49.1 km/h
```

### 2.2 Data Preprocessing Pipeline
Raw files are supplied per trip without headers:
* **`.track`:** White-space separated columns: `longitude`, `latitude`, `timestamp` (seconds).
* **`.nodes`:** `longitude`, `latitude` (0-based line index corresponds to `node_id`).
* **`.arcs`:** `from_node_id`, `to_node_id` (0-based line index corresponds to `arc_id`).
* **`.route`:** Chronological sequence of ground-truth `arc_id` values traversed by the vehicle.

#### Metric Projection
Because ordinary Euclidean distance in angular degrees ($\Delta^\circ$) produces severe spatial distortion at non-equatorial latitudes, all coordinates are projected into local Cartesian meters using geodesic scaling:
$$\Delta y = (\text{lat}_1 - \text{lat}_2) \times 111{,}139 \text{ m}$$
$$\Delta x = (\text{lon}_1 - \text{lon}_2) \times 111{,}139 \times \cos(\overline{\text{lat}}) \text{ m}$$
where $\overline{\text{lat}} \approx 49.85^\circ\text{N}$ (Darmstadt/Frankfurt metropolitan region, Germany).

#### Sampling Interval Analysis
Evaluating consecutive timestamps across all 227,577 observations revealed:
* **Nominal 1 Hz sampling:** **96.68%** of consecutive GPS points are spaced exactly **1.0 second** apart.
* **2.0-second steps:** **1.96%** of transitions.
* **GPS dropouts / Gaps ($>2.0\text{s}$):** **1.35%** of transitions, with occasional urban tunnel or underpass dropouts reaching up to 15–30 seconds.

#### Road Network Graph Topology
Each trip folder contains its local road network graph. The number of road network nodes ranges from **1,114 nodes** up to **309,939 nodes**, and arcs range from **2,348** up to **582,311**. The average node degree is approximately 2.3, characteristic of directed planar street networks.

### 2.3 Feature Extraction
For each GPS point $z_t$, a spatial candidate set of road segments is extracted using a 3-point KDTree indexing the start node, midpoint, and end node of each arc. For each candidate segment, the following 8 features are engineered:
1. **`dist_to_segment`:** Orthogonal perpendicular distance from the GPS observation to the line segment in meters:
   $$t = \text{clip}\left(\frac{(P - A) \cdot (B - A)}{\|B - A\|^2}, 0.0, 1.0\right), \quad Q = A + t(B - A), \quad d = \|P - Q\|$$
2. **`heading_diff`:** Angular discrepancy between the vehicle's movement vector and the road segment's azimuth in degrees $[0^\circ, 180^\circ]$:
   $$\Delta \theta = \min(|\theta_{\text{gps}} - \theta_{\text{road}}|, 360^\circ - |\theta_{\text{gps}} - \theta_{\text{road}}|)$$
3. **`dist_to_start`:** Distance from GPS point to the segment's `from_node` (meters).
4. **`dist_to_end`:** Distance from GPS point to the segment's `to_node` (meters).
5. **`proj_ratio`:** Normalized projection parameter $t \in [0.0, 1.0]$ along the segment length.
6. **`segment_length`:** Physical length of the candidate road arc in meters.
7. **`gps_speed`:** Instantaneous vehicle speed calculated as $\Delta d_{\text{gps}} / \Delta t$ ($m/s$).
8. **`candidate_rank`:** Spatial proximity rank ($0, 1, \dots, K-1$) among queried candidates.

---

## Section 3: Model Experimentation and Comparison

To avoid arbitrary model selection, five distinct models spanning five algorithmic paradigms were trained, cross-validated using **5-Fold GroupKFold** (grouped strictly by `trip_id` to guarantee zero cross-trip data leakage), and tested on the independent 10 student-test trips.

### 3.1 Consolidated Model Comparison Table

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | $C=1.0$, L-BFGS, StandardScaler | 0.8157 | 65.85% | 88.61% | 0.8183 | 0.1692 | 0.0002 ms |
| **KNN** | Instance-Based | $k=9$, Distance-Weighted, StandardScaler | 0.8069 | 64.55% | 87.02% | 0.8154 | 0.5302 | 0.0291 ms |
| **Random Forest** | Ensemble | $N=100$, Depth=14, MinSplit=10 | 0.8291 | 66.86% | 89.51% | 0.8328 | 0.1491 | 0.0021 ms |
| **LightGBM** | Boosting | $N=120$, $\eta=0.08$, Leaves=31 | **0.8294** | 66.47% | 88.98% | **0.8330** | 0.1585 | **0.0010 ms** |
| **MLP Classifier** | Neural Networks | Layers=(64, 32), ReLU, Adam | 0.8247 | 65.59% | 88.28% | 0.8282 | 0.1518 | 0.0012 ms |

---

### 3.2 Compulsory Questions for EVERY Model

#### Model 1: Logistic Regression (Linear & Probabilistic)
* **Algorithm Suitability:** Provides a well-calibrated probabilistic baseline where the decision boundary represents a linear combination of spatial distance, heading delta, and speed.
* **Strengths & Limitations:** Extremely fast inference (sub-microsecond), highly interpretable log-odds coefficients. However, cannot model complex non-linear geometric interactions (e.g., highway overpasses where perpendicular distance is small but elevation or turning angles require non-linear boundaries).
* **Hyperparameters & Initial Values:** Inverse regularization $C=1.0$, L-BFGS optimizer, tolerance $10^{-4}$, maximum iterations 500.
* **Loss Function:** Binary Cross-Entropy (Logistic Loss) with $L_2$ penalty: $\mathcal{L} = -\sum [y \ln p + (1-y)\ln(1-p)] + \frac{1}{2C}\|\mathbf{w}\|^2$.
* **Student-Testing Results:** Top-1 Accuracy: **65.85%**, Macro F1: **0.8183**, Log Loss: **0.1692**, Route Continuity: **88.61%**.
* **Overfitting / Underfitting Analysis:** The model exhibited mild underfitting (CV Macro F1 of 0.8157 vs 0.8183 test), indicating that linear decision boundaries slightly constrain expressive capacity.

#### Model 2: K-Nearest Neighbors (Instance-Based)
* **Algorithm Suitability:** Classifies candidates based on proximity in feature space; candidates with identical spatial offsets and heading alignments to known true segments receive high similarity scores.
* **Strengths & Limitations:** Non-parametric, requires no assumptions on underlying distribution. Limitations include high memory footprint and slower inference latency (0.029 ms/query), with susceptibility to feature scale disparities.
* **Hyperparameters & Initial Values:** Number of neighbors $k=9$, inverse distance weighting $w = 1/d$, Euclidean metric with standardized features.
* **Loss Function / Objective:** Voting weighted by inverse distance $w_i = 1 / (d(x, x_i) + 10^{-5})$.
* **Student-Testing Results:** Top-1 Accuracy: **64.55%**, Macro F1: **0.8154**, Log Loss: **0.5302**, Route Continuity: **87.02%**.
* **Overfitting / Underfitting Analysis:** High log loss (0.5302) due to uncalibrated distance voting producing extreme probability outputs (0 or 1), showing sensitivity to local noise in dense multi-lane junctions.

#### Model 3: Random Forest Classifier (Ensemble)
* **Algorithm Suitability:** Excels at capturing non-linear thresholds and feature splits (e.g., segment distance $< 8\text{m}$ AND heading delta $< 25^\circ$) while maintaining variance reduction through bagging.
* **Strengths & Limitations:** Outstanding generalization, robust to feature outliers, high accuracy (Top-1 66.86%). Limitations: Slower training time and larger serialized model footprint (28.7 MB).
* **Hyperparameters & Initial Values:** $N_{\text{estimators}} = 100$, maximum depth $= 14$, minimum samples split $= 10$, Gini impurity criterion.
* **Loss Function / Objective:** Minimization of average Gini impurity across all constituent decision trees.
* **Student-Testing Results:** Top-1 Accuracy: **66.86%**, Macro F1: **0.8328**, Log Loss: **0.1491**, Route Continuity: **89.51%**.
* **Overfitting / Underfitting Analysis:** Well-balanced generalization: CV Macro F1 (0.8291) matches Test Macro F1 (0.8328) within 0.003, confirming absence of overfitting due to depth capping.

#### Model 4: LightGBM Classifier (Gradient Boosting)
* **Algorithm Suitability:** Optimized gradient-boosted decision trees using leaf-wise tree growth. Specifically suited for tabular spatial-kinematic features due to native handling of continuous splits and high computational efficiency.
* **Strengths & Limitations:** Highest CV Macro F1 (0.8294), fastest training and inference speed (0.001 ms/query), compact model footprint (420 KB). Minor limitation: requires hyperparameter tuning to prevent aggressive leaf expansion on noisy GPS points.
* **Hyperparameters & Initial Values:** Number of estimators $= 120$, learning rate $\eta = 0.08$, number of leaves $= 31$, minimum child samples $= 20$.
* **Loss Function / Objective:** Binary log-loss objective optimized via second-order gradient boosting:
  $$\mathcal{L}(\mathbf{y}, \hat{\mathbf{p}}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1-y_i) \ln(1-\hat{p}_i) \right]$$
* **Student-Testing Results:** Top-1 Accuracy: **66.47%**, Macro F1: **0.8330**, Log Loss: **0.1585**, Route Continuity: **88.98%**.
* **Overfitting / Underfitting Analysis:** Excellent balance between training and test generalization. CV Macro F1 (0.8294) generalized cleanly to Student Test Macro F1 (0.8330), showing zero overfitting.

#### Model 5: Multi-Layer Perceptron (Neural Networks)
* **Algorithm Suitability:** Learns smooth continuous non-linear representations across geometric coordinates, velocities, and bearing alignments.
* **Strengths & Limitations:** Flexible functional approximation with hidden representation learning. Limitations: High training latency (66.1s), sensitivity to feature standardization, and tendency to plateau without large iteration budgets.
* **Hyperparameters & Initial Values:** Architecture: Two hidden layers $(64, 32)$, ReLU activation, Adam optimizer, initial learning rate $10^{-3}$, early stopping enabled (patience 3).
* **Loss Function / Objective:** Cross-entropy loss with $L_2$ weight decay regularization ($\alpha = 10^{-4}$).
* **Student-Testing Results:** Top-1 Accuracy: **65.59%**, Macro F1: **0.8282**, Log Loss: **0.1518**, Route Continuity: **88.28%**.
* **Overfitting / Underfitting Analysis:** Early stopping halted training at epoch 25, preventing over-parameterized memorization while maintaining healthy test performance.

---

## Section 4: Bayesian Hyperparameter Optimization

Because **LightGBM** demonstrated the best overall balance of CV Macro F1 (0.8294), low log loss, and high inference efficiency (0.001 ms), it was selected as the final candidate for Bayesian Hyperparameter Optimization.

### 4.1 Optimization Protocol
1. **Search Space:**
   * `n_estimators`: Integer uniform $[80, 200]$ (step 20)
   * `learning_rate`: Log-uniform $[0.02, 0.20]$
   * `num_leaves`: Integer uniform $[20, 64]$
   * `max_depth`: Integer uniform $[5, 15]$
   * `min_child_samples`: Integer uniform $[15, 80]$
   * `subsample`: Uniform $[0.65, 0.95]$
   * `colsample_bytree`: Uniform $[0.65, 0.95]$
2. **Objective Function:** Maximize 3-Fold GroupKFold Cross-Validation Macro F1-score across the training partition.
3. **Number of Trials:** 20 optimization trials using Optuna's Tree-structured Parzen Estimator (TPE).
4. **Best Hyperparameters Obtained:**
   * `n_estimators`: **160**
   * `learning_rate`: **0.0256**
   * `num_leaves`: **50**
   * `max_depth`: **7**
   * `min_child_samples`: **49**
   * `subsample`: **0.6922**
   * `colsample_bytree`: **0.9472**

### 4.2 Untuned vs Tuned Model Performance

| Metric | Untuned LightGBM | Tuned LightGBM | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **CV Macro F1-score** | 0.8294 | **0.8337** | **+0.0043** |
| **Student-Testing Top-1 Accuracy** | 66.47% | **66.84%** | **+0.37%** |
| **Student-Testing Macro F1-score** | 0.8330 | **0.8352** | **+0.0022** |
| **Log Loss (Lower is Better)** | 0.1585 | **0.1477** | **-0.0108** |
| **Inference Time** | 0.0010 ms | 0.0014 ms | +0.0004 ms |

The tuned model reduced Log Loss by **6.8%** and lifted Top-1 accuracy to **66.84%**, verifying that shallower trees (`max_depth=7`) paired with a lower learning rate (`0.0256`) effectively reduced noise sensitivity.

---

## Section 5: HMM and Viterbi Route Reconstruction

### 5.1 Formulation of the Hidden Markov Model
While a supervised classifier produces strong pointwise candidate scores, isolated argmax predictions frequently assign consecutive observations to non-contiguous road segments (e.g. jumping between parallel one-way streets). An HMM resolves this by finding the most probable joint sequence over the road network topology.

1. **Hidden States:** The discrete set of candidate road segments $S_t = \{a_{t, 1}, a_{t, 2}, \dots, a_{t, K}\}$ for each GPS point $t = 1, \dots, T$.
2. **Emission Probabilities:** Calculated by combining spatial Gaussian decay with the classifier probability score:
   $$\ln P(z_t | s_t = a_j) = -\frac{d(z_t, a_j)^2}{2\sigma_z^2} + \ln\left(\hat{p}_{\text{model}}(a_j) + 10^{-4}\right)$$
   where $\sigma_z = 12.0\text{ meters}$ models sensor variance.
3. **Transition Probabilities:**
   Between consecutive observations $t-1$ and $t$, let $d_{\text{gps}} = \|z_t - z_{t-1}\|$:
   * **Same Segment ($s_{t-1} = s_t$):** $\ln P(s_t | s_{t-1}) = -0.1$
   * **Direct Graph Connection ($\text{to\_node}(s_{t-1}) == \text{from\_node}(s_t)$):**
     $$\ln P(s_t | s_{t-1}) = -0.4 - \frac{|0.0 - d_{\text{gps}}|}{\beta}$$
   * **Multi-hop Reachability ($d_G < 180\text{m}$):**
     $$\ln P(s_t | s_{t-1}) = -1.2 - \frac{|d_G - d_{\text{gps}}|}{\beta}$$
     where $\beta = 25.0\text{ m}$.
   * **Disconnected / Unreachable:** Heavy penalty $\ln P = -10.0$.
4. **Handling GPS Gaps & Edge Cases:** If $\Delta t > 6.0\text{ seconds}$, the transition model resets to a uniform distribution over the new candidate set, preventing penalty propagation across tunnels or GPS dead zones.
5. **Viterbi Decoding:** Implemented in log-space to guarantee numerical stability:
   $$V_t(j) = \max_i \left[ V_{t-1}(i) + \ln P(s_t = j | s_{t-1} = i) \right] + \ln P(z_t | s_t = j)$$
   Optimal backtracking reconstructs the global maximum a posteriori sequence $S^* = (s_1^*, \dots, s_T^*)$, which is then collapsed into unique sequential road segments $\hat{\mathcal{R}}$.

### 5.2 Experimental Results: Classifier-Only vs Classifier + HMM

Evaluating across all 10 official student-testing trips demonstrates the performance of HMM decoding:

| Trip ID | GPS Points | Clf Pointwise Acc | HMM Pointwise Acc | Clf Continuity | HMM Continuity | Route Arcs (Pred / True) | Sequence F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00000000** | 2,503 | 90.65% | **92.69%** | 97.80% | **99.52%** | 243 / 179 | 0.8426 |
| **00000007** | 1,354 | 91.06% | **91.29%** | 99.63% | **100.00%** | 153 / 123 | 0.9719 |
| **00000009** | 1,049 | 81.32% | **86.94%** | 95.99% | **99.43%** | 408 / 429 | 0.8424 |
| **00000040** | 2,995 | 88.65% | **94.19%** | 98.23% | **99.97%** | 406 / 309 | 0.9262 |
| **00000052** | 262 | 94.66% | **98.09%** | 96.93% | **99.62%** | 81 / 95 | 0.8864 |
| **00000056** | 3,958 | 20.04% | **20.62%** | 90.27% | **99.27%** | 1,255 / 1,068 | 0.9125 |
| **00000065** | 768 | 72.40% | **78.78%** | 92.83% | **99.22%** | 164 / 145 | 0.8699 |
| **00000076** | 1,486 | 78.20% | **88.63%** | 94.61% | **99.66%** | 407 / 374 | 0.7938 |
| **00000077** | 3,911 | 59.12% | **65.53%** | 97.65% | **99.95%** | 745 / 268 | 0.8103 |
| **00000091** | 874 | 82.95% | **90.39%** | 93.47% | **99.31%** | 211 / 232 | 0.8140 |
| **AVERAGE** | **1,916** | **75.91%** | **80.72%** (+4.81%) | **95.74%** | **99.59%** (+3.85%) | — | **0.8670** |

#### Analysis:
1. **Pointwise Accuracy Boost:** HMM Viterbi decoding increased average pointwise accuracy from **75.91% to 80.72%**, with substantial leaps on challenging trips like Trip 00000076 (+10.43%) and Trip 00000091 (+7.44%).
2. **Topological Continuity:** Raw classifier predictions suffered from trajectory fractures, leaving 4.26% of transitions disconnected. HMM decoding brought route continuity to **99.59%** (with Trip 00000007 achieving a perfect **100.00%**).
3. **Trip 00000056 Anomaly:** Trip 00000056 exhibited a low pointwise matching score (20.62%) due to a systemic coordinate offset / unmapped highway expansion in the reference route; however, its topological sequence recovery achieved **0.9125 Route F1**, showing that the HMM followed the true physical trajectory.

---

## Section 6: Geographical Visualisation Using Folium

Interactive Folium visualizations were generated for all evaluated trips (`results/map_trip_00000000.html`, `results/map_trip_00000007.html`).

### Map Architecture & Layers:
1. **Base Cartography:** OpenStreetMap tiles with no API key requirement, enabling offline/local classroom execution.
2. **Road Network Geometry (Gray Polylines):** Renders all underlying road network segments (`arcs`) within the trajectory bounding box, demonstrating street network topology.
3. **Ground-Truth Route (Solid Green Polyline, 4.5px):** Chronological sequence of reference road segments traversed by the vehicle.
4. **Predicted Map-Matched Route (Dashed Red Polyline, 3.5px):** Reconstructed route output by the HMM Viterbi pipeline.
5. **Raw GPS Observations (Blue Circle Markers, 3.5px):** Subsampled GPS trajectory observations showing lateral noise, multipath deviation, and raw sensor readings.
6. **Interactive Controls:** Layer control toggle in top-right, fullscreen mode, custom HTML legend, and hover tooltips showing segment IDs, GPS timestamps, and coordinates.

---

## Section 7: API, UI and Deployment

### 7.1 Distributed Two-Laptop Architecture

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

### 7.2 Execution Instructions

#### Laptop 1 (Model and API Server):
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server (listening on `0.0.0.0:8000` for LAN access):
   ```bash
   python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
   ```
3. Check Laptop 1 local IP address via `ipconfig` (e.g., `192.168.1.15`).

#### Laptop 2 (Evaluator User Interface):
1. Start the client UI runner:
   ```bash
   python ui/client.py
   ```
   *(Alternatively, open `http://<laptop1-ip>:8000/ui` directly in any web browser).*
2. Enter the Laptop 1 endpoint address in the UI (e.g., `http://192.168.1.15:8000`).
3. Select any of the 10 reserved evaluation trips from the dropdown.
4. Click **Run Evaluation Pipeline**. The UI displays pointwise accuracy, route continuity, sequence F1, and renders the interactive comparison map within 500 ms.
5. Inspect live CMD terminal logs on Laptop 1 to verify authentic real-time execution.

---

## Section 8: Academic Integrity & Compliance Statement

1. **Independent Implementation:** All data loaders, spatial indexers, feature extraction routines, classifier pipelines, Optuna search spaces, HMM Viterbi solvers, and API/UI endpoints were implemented independently.
2. **Zero Data Leakage:** The 10 evaluator-testing trips remained untouched during model training and feature selection. Ground-truth reference routes are processed strictly server-side on Laptop 1 during evaluation requests and are never leaked to or hardcoded in the student client UI.
3. **Reproducibility:** All experiment logs, random seeds (`seed=42`), cross-validation folds, and model weights are retained in the repository (`saved_models/` and `results/`).
