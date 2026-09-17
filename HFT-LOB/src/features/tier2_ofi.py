"""
Tier 2: Order Flow Imbalance (OFI) Features.
Implements Equations (5) - (7) and (12) - (14):
- Piecewise level-k OFI Delta W_k,t^b - Delta W_k,t^a
- Multi-level OFI vector OFI_t in R^K
- Integrated OFI via PCA first component OFI_t^int = u_1^T OFI_t
- Cumulative Imbalance Index (CII_t = sum_{tau=t-W}^t OFI_tau^int)
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.math.formulas import compute_level_ofi


def extract_tier2_ofi(
    df: pd.DataFrame,
    num_levels: int = 10,
    cii_windows: List[int] = [5, 10, 20],
    pca_component_vec: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Extracts Tier 2 Order Flow Imbalance features and computes integrated PCA projection.
    """
    t2_df = pd.DataFrame(index=df.index)
    ofi_matrix = np.zeros((len(df), num_levels), dtype=np.float64)
    
    for k in range(1, num_levels + 1):
        bp_curr = df[f"bid_p_{k}"].values
        bp_prev = df[f"bid_p_{k}"].shift(1).fillna(df[f"bid_p_{k}"].iloc[0]).values
        bv_curr = df[f"bid_v_{k}"].values
        bv_prev = df[f"bid_v_{k}"].shift(1).fillna(df[f"bid_v_{k}"].iloc[0]).values
        
        ap_curr = df[f"ask_p_{k}"].values
        ap_prev = df[f"ask_p_{k}"].shift(1).fillna(df[f"ask_p_{k}"].iloc[0]).values
        av_curr = df[f"ask_v_{k}"].values
        av_prev = df[f"ask_v_{k}"].shift(1).fillna(df[f"ask_v_{k}"].iloc[0]).values
        
        ofi_k = compute_level_ofi(
            bp_curr, bp_prev, bv_curr, bv_prev,
            ap_curr, ap_prev, av_curr, av_prev
        )
        ofi_matrix[:, k-1] = ofi_k
        t2_df[f"ofi_level_{k}"] = ofi_k

    # Best-level OFI
    t2_df["ofi_best_level"] = ofi_matrix[:, 0]
    
    # Estimate or apply PCA projection u_1
    if pca_component_vec is None:
        pca = PCA(n_components=1)
        pca.fit(ofi_matrix)
        pca_component_vec = pca.components_[0]
        # Align sign: positive OFI should correspond to upward price pressure
        if np.mean(pca_component_vec) < 0:
            pca_component_vec = -pca_component_vec

    # Equation (13): Integrated OFI
    ofi_integrated = np.dot(ofi_matrix, pca_component_vec)
    t2_df["ofi_integrated"] = ofi_integrated
    
    # Equation (14): Cumulative Imbalance Index (CII)
    for w in cii_windows:
        t2_df[f"cii_window_{w}"] = pd.Series(ofi_integrated).rolling(window=w, min_periods=1).sum().values

    return t2_df, list(t2_df.columns), pca_component_vec
