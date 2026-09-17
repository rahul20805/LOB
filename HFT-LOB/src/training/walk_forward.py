"""
Walk-Forward Time-Series Cross-Validation Engine (Section 7.2 & RAEP Protocol).
Strictly chronological rolling windows. Zero random shuffling.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class WalkForwardSplitter:
    """
    Generates rolling walk-forward fold indices for strictly chronological time series.
    """
    def __init__(self, num_folds: int = 5, train_window_folds: int = 2, val_fraction: float = 0.2):
        self.num_folds = num_folds
        self.train_window_folds = train_window_folds
        self.val_fraction = val_fraction

    def split(self, total_samples: int) -> List[Dict[str, np.ndarray]]:
        """
        Partitions total_samples into M chronological folds and builds rolling train/val/test indices.
        """
        fold_size = total_samples // self.num_folds
        folds_indices = []
        for m in range(self.num_folds):
            start = m * fold_size
            end = (m + 1) * fold_size if m < self.num_folds - 1 else total_samples
            folds_indices.append(np.arange(start, end))

        splits = []
        # Walk-forward evaluation on test folds
        for test_m in range(1, self.num_folds):
            # Training folds: up to train_window_folds prior
            start_train_m = max(0, test_m - self.train_window_folds)
            train_folds = folds_indices[start_train_m : test_m]
            train_pool = np.concatenate(train_folds)
            
            # Split train_pool into train and validation
            n_val = int(len(train_pool) * self.val_fraction)
            train_idx = train_pool[:-n_val]
            val_idx = train_pool[-n_val:]
            test_idx = folds_indices[test_m]
            
            splits.append({
                "fold": test_m,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx
            })
            
        return splits
