"""
Feature Engine unifying Tiers 0, 1, 2, and 3 into standardized feature matrices.
Also builds sequence tensors for Deep Learning architectures (CNN, LSTM, Transformer).
"""

from typing import Tuple, List, Dict, Optional, Union
import os
import numpy as np
import pandas as pd
from src.features.tier0_raw import extract_tier0_raw_lob
from src.features.tier1_microstructure import extract_tier1_microstructure
from src.features.tier2_ofi import extract_tier2_ofi
from src.features.tier3_dynamics import extract_tier3_dynamics


class FeatureEngine:
    """
    Standardized feature extraction and sequence formatting pipeline.
    """
    def __init__(self, num_levels: int = 10, sequence_length: int = 20):
        self.num_levels = num_levels
        self.sequence_length = sequence_length
        self.pca_component_vec: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.feature_tiers: Dict[str, str] = {}

    def extract_all_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Extracts Tier 0, Tier 1, Tier 2, and Tier 3 features.
        """
        # Tier 0: Raw LOB
        t0_arr, t0_names = extract_tier0_raw_lob(df, num_levels=self.num_levels)
        t0_df = pd.DataFrame(t0_arr, columns=t0_names, index=df.index)
        
        # Tier 1: Microstructure
        t1_df, t1_names = extract_tier1_microstructure(df, num_levels=self.num_levels)
        
        # Tier 2: OFI
        pca_vec = None if is_training else self.pca_component_vec
        t2_df, t2_names, fitted_pca = extract_tier2_ofi(
            df, num_levels=self.num_levels, pca_component_vec=pca_vec
        )
        if is_training:
            self.pca_component_vec = fitted_pca

        # Tier 3: Dynamics
        t3_df, t3_names = extract_tier3_dynamics(df)
        
        # Combine
        combined_df = pd.concat([t0_df, t1_df, t2_df, t3_df], axis=1)
        self.feature_names = list(combined_df.columns)
        
        tier_mapping = {
            "Tier 0 (Raw LOB)": t0_names,
            "Tier 1 (Microstructure)": t1_names,
            "Tier 2 (OFI)": t2_names,
            "Tier 3 (Dynamics)": t3_names
        }
        
        for tier, cols in tier_mapping.items():
            for c in cols:
                self.feature_tiers[c] = tier

        return combined_df, tier_mapping

    def export_feature_dictionary(self, filepath: str = "reports/feature_dictionary.csv") -> str:
        """
        Generates and saves comprehensive feature dictionary report.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        records = []
        for feat in self.feature_names:
            tier = self.feature_tiers.get(feat, "Unknown")
            if "p_" in feat:
                desc = "Limit order book price level"
                formula_ref = "Eq (1)"
            elif "v_" in feat:
                desc = "Limit order book volume depth"
                formula_ref = "Eq (1)"
            elif "spread" in feat:
                desc = "Bid-ask spread metric"
                formula_ref = "Eq (8)"
            elif "microprice" in feat:
                desc = "Volume-weighted mid price"
                formula_ref = "Eq (11)"
            elif "imbalance" in feat:
                desc = "Order book queue/volume imbalance"
                formula_ref = "Eq (10)"
            elif "ofi" in feat:
                desc = "Order flow imbalance metric"
                formula_ref = "Eq (5)-(7), (12)-(13)"
            elif "cii" in feat:
                desc = "Cumulative imbalance index"
                formula_ref = "Eq (14)"
            elif "momentum" in feat or "return" in feat:
                desc = "Multi-horizon price trend indicator"
                formula_ref = "Sec 4.2.4"
            elif "vol" in feat:
                desc = "Realized tick volatility"
                formula_ref = "Sec 4.2.4"
            elif "time" in feat:
                desc = "Intraday cyclical time encoding"
                formula_ref = "Sec 4.2.4"
            else:
                desc = "Microstructural indicator"
                formula_ref = "Taxonomy"

            records.append({
                "feature_name": feat,
                "tier": tier,
                "description": desc,
                "formula_reference": formula_ref,
                "data_type": "float64"
            })
            
        dict_df = pd.DataFrame(records)
        dict_df.to_csv(filepath, index=False)
        return filepath

    def build_sequence_tensors(
        self,
        feature_matrix: np.ndarray,
        target_labels: np.ndarray,
        seq_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts 2D tabular features into 3D (samples, seq_len, features) for PyTorch CNN/LSTM/Transformer.
        """
        if seq_len is None:
            seq_len = self.sequence_length
            
        N, D = feature_matrix.shape
        if N <= seq_len:
            raise ValueError(f"Insufficient samples {N} for sequence length {seq_len}")
            
        num_windows = N - seq_len + 1
        X_seq = np.zeros((num_windows, seq_len, D), dtype=np.float32)
        y_seq = np.zeros(num_windows, dtype=np.int64)
        
        for i in range(num_windows):
            X_seq[i] = feature_matrix[i : i + seq_len]
            y_seq[i] = target_labels[i + seq_len - 1]
            
        return X_seq, y_seq
