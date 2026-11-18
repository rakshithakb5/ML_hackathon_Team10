"""
PathSnap: High-Performance Hidden Markov Model (HMM) & Viterbi Route Reconstruction
Optimized with local topological graph caching:
- Hidden States: Candidate road segments
- Emission Probabilities: Perpendicular distance Gaussian likelihood + Classifier confidence
- Transition Probabilities: Ultra-fast local adjacency BFS (1-hop, 2-hop, 3-hop) vs GPS displacement
- Viterbi Algorithm: Dynamic programming in log-space
- Edge Cases: Handles GPS gaps (>6s) and network discontinuities
"""

import os
import sys
import numpy as np
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.loader import TripDataLoader
from features.candidate_extractor import CandidateExtractor
from models.train_all import FEATURE_COLS

class HMMMapMatcher:
    """Ultra-fast HMM & Viterbi Route Reconstructor."""
    
    def __init__(
        self,
        trip: TripDataLoader,
        classifier: Any,
        sigma_z: float = 12.0,
        beta: float = 25.0,
        gap_threshold_s: float = 6.0
    ):
        self.trip = trip
        self.classifier = classifier
        self.sigma_z = sigma_z
        self.beta = beta
        self.gap_threshold_s = gap_threshold_s
        
        # Build ultra-fast adjacency dictionary: node -> [(neighbor, weight)]
        self.adj = defaultdict(list)
        nodes_m = self.trip.to_meters(self.trip.nodes)
        
        for arc_id, (u, v) in enumerate(self.trip.arcs):
            p1 = nodes_m[u]
            p2 = nodes_m[v]
            length = float(np.sqrt(np.sum((p2 - p1)**2)))
            self.adj[int(u)].append((int(v), length))
            
        self._cache = {}

    def get_local_distance(self, u: int, v: int) -> float:
        """Finds shortest road distance from node u to node v within a fast 3-hop horizon."""
        if u == v:
            return 0.0
            
        cache_key = (u, v)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # 1-hop check
        for n1, d1 in self.adj.get(u, []):
            if n1 == v:
                self._cache[cache_key] = d1
                return d1
                
        # 2-hop check
        min_2hop = float('inf')
        for n1, d1 in self.adj.get(u, []):
            for n2, d2 in self.adj.get(n1, []):
                if n2 == v:
                    d = d1 + d2
                    if d < min_2hop:
                        min_2hop = d
        if min_2hop < float('inf'):
            self._cache[cache_key] = min_2hop
            return min_2hop
            
        # 3-hop check
        min_3hop = float('inf')
        for n1, d1 in self.adj.get(u, []):
            for n2, d2 in self.adj.get(n1, []):
                for n3, d3 in self.adj.get(n2, []):
                    if n3 == v:
                        d = d1 + d2 + d3
                        if d < min_3hop:
                            min_3hop = d
        if min_3hop < float('inf'):
            self._cache[cache_key] = min_3hop
            return min_3hop
            
        self._cache[cache_key] = float('inf')
        return float('inf')

    def run_reconstruction(self, extractor: CandidateExtractor) -> Dict[str, Any]:
        """Runs fast Viterbi decoding over all GPS observations."""
        T = len(extractor.track_m)
        if T == 0:
            return {}
            
        # 1. Batch Candidate Extraction and Classification
        candidates_by_step: List[List[Dict[str, Any]]] = []
        all_features = []
        step_lengths = []
        
        for t in range(T):
            cands = extractor.extract_candidates_for_point(t)
            if len(cands) == 0:
                cands = [{
                    "arc_id": 0, "from_node": 0, "to_node": 0,
                    "dist_to_segment": 50.0, "is_match": 0
                }]
            candidates_by_step.append(cands)
            step_lengths.append(len(cands))
            for c in cands:
                all_features.append([c.get(col, 0.0) for col in FEATURE_COLS])
                
        # Batch predict all candidate probabilities in a single vectorized call
        all_features = np.array(all_features, dtype=np.float32)
        try:
            all_probs = self.classifier.predict_proba(all_features)[:, 1]
        except Exception:
            all_probs = np.full(len(all_features), 0.1, dtype=np.float32)
            
        offset = 0
        classifier_top1 = np.zeros(T, dtype=np.int64)
        for t in range(T):
            k = step_lengths[t]
            probs_t = all_probs[offset:offset + k]
            for j in range(k):
                candidates_by_step[t][j]["prob"] = float(probs_t[j])
            best_idx = int(np.argmax(probs_t))
            classifier_top1[t] = candidates_by_step[t][best_idx]["arc_id"]
            offset += k
            
        # 2. Viterbi Trellis in Log-Space
        V = [np.full(len(candidates_by_step[t]), -np.inf) for t in range(T)]
        Backpointers = [np.zeros(len(candidates_by_step[t]), dtype=np.int32) for t in range(T)]
        
        # t = 0 initialization
        for j, cand in enumerate(candidates_by_step[0]):
            d = cand["dist_to_segment"]
            log_emission = -0.5 * (d / self.sigma_z)**2 + np.log(cand["prob"] + 1e-4)
            V[0][j] = log_emission
            
        # Forward pass
        for t in range(1, T):
            dt = float(extractor.timestamps[t] - extractor.timestamps[t-1])
            is_gap = dt > self.gap_threshold_s
            
            p_curr = extractor.track_m[t]
            p_prev = extractor.track_m[t-1]
            d_gps = float(np.sqrt(np.sum((p_curr - p_prev)**2)))
            
            curr_cands = candidates_by_step[t]
            prev_cands = candidates_by_step[t-1]
            
            for j, c_curr in enumerate(curr_cands):
                d = c_curr["dist_to_segment"]
                log_emission = -0.5 * (d / self.sigma_z)**2 + np.log(c_curr["prob"] + 1e-4)
                
                best_prev_val = -np.inf
                best_prev_idx = 0
                
                for i, c_prev in enumerate(prev_cands):
                    if is_gap:
                        log_trans = -np.log(len(curr_cands))
                    else:
                        arc_prev = c_prev["arc_id"]
                        arc_curr = c_curr["arc_id"]
                        
                        if arc_prev == arc_curr:
                            log_trans = -0.1
                        elif c_prev["to_node"] == c_curr["from_node"]:
                            diff = abs(0.0 - d_gps)
                            log_trans = -0.4 - (diff / self.beta)
                        else:
                            u = c_prev["to_node"]
                            v = c_curr["from_node"]
                            net_dist = self.get_local_distance(u, v)
                            if net_dist < 180.0:
                                log_trans = -1.2 - (abs(net_dist - d_gps) / self.beta)
                            else:
                                log_trans = -10.0
                                
                    score = V[t-1][i] + log_trans
                    if score > best_prev_val:
                        best_prev_val = score
                        best_prev_idx = i
                        
                V[t][j] = best_prev_val + log_emission
                Backpointers[t][j] = best_prev_idx
                
        # 3. Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        predicted_states = np.zeros(T, dtype=np.int64)
        curr_idx = best_last_idx
        
        for t in range(T - 1, -1, -1):
            predicted_states[t] = candidates_by_step[t][curr_idx]["arc_id"]
            if t > 0:
                curr_idx = Backpointers[t][curr_idx]
                
        # 4. Collapse sequence into reconstructed route
        reconstructed_route = []
        for arc in predicted_states:
            if len(reconstructed_route) == 0 or reconstructed_route[-1] != arc:
                reconstructed_route.append(int(arc))
                
        # 5. Compute Metrics
        true_arcs = extractor.point_true_arcs
        obs_acc_clf = float(np.mean(classifier_top1 == true_arcs) * 100.0) if true_arcs is not None else 0.0
        obs_acc_hmm = float(np.mean(predicted_states == true_arcs) * 100.0) if true_arcs is not None else 0.0
        
        continuity_clf = self._calc_continuity(classifier_top1)
        continuity_hmm = self._calc_continuity(predicted_states)
        
        return {
            "predicted_states": predicted_states,
            "classifier_states": classifier_top1,
            "reconstructed_route": reconstructed_route,
            "observation_acc_clf": round(obs_acc_clf, 2),
            "observation_acc_hmm": round(obs_acc_hmm, 2),
            "continuity_clf": round(continuity_clf, 2),
            "continuity_hmm": round(continuity_hmm, 2)
        }

    def _calc_continuity(self, state_sequence: np.ndarray) -> float:
        """Calculates percentage of consecutive transitions that connect in the road network."""
        T = len(state_sequence)
        if T <= 1:
            return 100.0
            
        connected = 0
        total = T - 1
        for i in range(total):
            a1 = state_sequence[i]
            a2 = state_sequence[i+1]
            if a1 == a2:
                connected += 1
            else:
                to_node = self.trip.arcs[a1, 1]
                from_node = self.trip.arcs[a2, 0]
                if to_node == from_node:
                    connected += 1
                else:
                    d = self.get_local_distance(int(to_node), int(from_node))
                    if d < 150.0:
                        connected += 1
                        
        return (connected / total) * 100.0
