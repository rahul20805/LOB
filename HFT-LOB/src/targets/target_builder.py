"""
Target construction module implementing Section 2.2 & Equations (3)-(4).
Builds forward smoothed mid-price return labels with horizon Delta and threshold alpha.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.math.formulas import compute_smoothed_return, assign_ternary_labels


class TargetBuilder:
    """
    Constructs ternary and continuous regression prediction targets from LOB mid-prices.
    """
    def __init__(self, horizon: int = 10, smoothing_window: int = 5, threshold_alpha: float = 0.0001):
        self.horizon = horizon
        self.smoothing_window = smoothing_window
        self.threshold_alpha = threshold_alpha

    def build_targets(self, mid_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds relative return l_t, ternary label y_t in {-1, 0, 1}, and 0-indexed class label {0, 1, 2}.
        Class 0: Down (-1)
        Class 1: Stationary (0)
        Class 2: Up (+1)
        """
        rel_returns = compute_smoothed_return(
            mid_prices=mid_prices,
            horizon=self.horizon,
            smoothing_window=self.smoothing_window
        )
        ternary_labels = assign_ternary_labels(
            relative_returns=rel_returns,
            threshold=self.threshold_alpha
        )
        
        # 0-indexed classes for PyTorch CrossEntropyLoss / Scikit-learn
        class_labels = np.zeros_like(ternary_labels, dtype=np.int64)
        class_labels[ternary_labels == -1] = 0 # Down
        class_labels[ternary_labels == 0] = 1  # Stationary
        class_labels[ternary_labels == 1] = 2  # Up

        return rel_returns, ternary_labels, class_labels

    def compute_class_distribution(self, class_labels: np.ndarray) -> Dict[str, float]:
        """
        Computes empirical class balance percentages.
        """
        total = len(class_labels)
        if total == 0:
            return {"down_pct": 0.0, "stat_pct": 0.0, "up_pct": 0.0}
            
        c0 = np.sum(class_labels == 0) / total
        c1 = np.sum(class_labels == 1) / total
        c2 = np.sum(class_labels == 2) / total
        
        return {
            "down_pct": float(c0 * 100),
            "stationary_pct": float(c1 * 100),
            "up_pct": float(c2 * 100)
        }
