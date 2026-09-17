"""
Dynamic Sharpe-Weighted Ensembles and Multi-Dataset Fusion.
Implements Section 7.4, 9.3 & Wong & Barahona (2023).
Dynamically weights constituent models based on their rolling Sharpe ratio in the active regime.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class DynamicSharpeEnsemble:
    """
    Dynamic model ensembling based on recent out-of-sample Sharpe performance.
    Down-weights poorly performing models in real-time during regime shifts.
    """
    def __init__(self, model_names: List[str], decay_factor: float = 0.85):
        self.model_names = model_names
        self.decay_factor = decay_factor
        self.weights = {m: 1.0 / len(model_names) for m in model_names}
        self.rolling_returns = {m: [] for m in model_names}

    def update_performance(self, model_returns: Dict[str, float]):
        """
        Updates trailing Sharpe ratio and adjusts model weights.
        """
        sharpes = {}
        for m in self.model_names:
            ret = model_returns.get(m, 0.0)
            self.rolling_returns[m].append(ret)
            if len(self.rolling_returns[m]) > 50:
                self.rolling_returns[m].pop(0)
                
            arr = np.array(self.rolling_returns[m])
            if len(arr) >= 3 and np.std(arr) > 1e-6:
                sh = np.mean(arr) / np.std(arr)
            else:
                # Early estimate based on mean return
                sh = 1.0 + np.mean(arr) * 10.0
            sharpes[m] = max(0.01, float(sh))

        # Softmax / normalized weighting based on Sharpe
        total_sh = sum(sharpes.values())
        for m in self.model_names:
            new_w = sharpes[m] / total_sh
            self.weights[m] = self.decay_factor * self.weights[m] + (1 - self.decay_factor) * new_w

    def predict_proba(self, model_probas: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes weighted probability ensemble.
        """
        combined_prob = None
        for m in self.model_names:
            prob = model_probas[m]
            w = self.weights[m]
            if combined_prob is None:
                combined_prob = w * prob
            else:
                combined_prob += w * prob
        return combined_prob
