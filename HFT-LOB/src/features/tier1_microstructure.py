"""
Tier 1: Basic Microstructure Features.
Implements Equations (8) - (11):
- Bid-ask spread s_t
- Mid-price m_t
- Multi-level volume imbalance psi_t
- Volume-weighted mid-price (microprice) m_t^w
- Level-1 depth ratio and top-K depth ratio
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from src.math.formulas import (
    compute_spread,
    compute_mid_price,
    compute_volume_imbalance,
    compute_weighted_mid_price
)


def extract_tier1_microstructure(df: pd.DataFrame, num_levels: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts Tier 1 microstructure features.
    """
    t1_df = pd.DataFrame(index=df.index)
    
    # 1. Best level prices and volumes
    ask_p1 = df["ask_p_1"].values
    bid_p1 = df["bid_p_1"].values
    ask_v1 = df["ask_v_1"].values
    bid_v1 = df["bid_v_1"].values
    
    # 2. Spread and mid
    t1_df["spread"] = compute_spread(ask_p1, bid_p1)
    t1_df["mid_price"] = compute_mid_price(ask_p1, bid_p1)
    t1_df["spread_bps"] = (t1_df["spread"] / t1_df["mid_price"]) * 10000.0
    
    # 3. Weighted mid-price (microprice)
    t1_df["microprice"] = compute_weighted_mid_price(bid_p1, bid_v1, ask_p1, ask_v1)
    t1_df["microprice_spread_dev"] = (t1_df["microprice"] - t1_df["mid_price"]) / (t1_df["spread"] + 1e-6)
    
    # 4. Multi-level Volume Imbalance (Eq 10)
    bid_v_cols = [f"bid_v_{k}" for k in range(1, num_levels + 1) if f"bid_v_{k}" in df.columns]
    ask_v_cols = [f"ask_v_{k}" for k in range(1, num_levels + 1) if f"ask_v_{k}" in df.columns]
    
    bid_v_matrix = df[bid_v_cols].values
    ask_v_matrix = df[ask_v_cols].values
    
    t1_df["volume_imbalance_topK"] = compute_volume_imbalance(bid_v_matrix, ask_v_matrix)
    t1_df["volume_imbalance_level1"] = compute_volume_imbalance(bid_v_matrix[:, :1], ask_v_matrix[:, :1])
    
    # 5. Total Depth
    t1_df["total_bid_depth"] = np.sum(bid_v_matrix, axis=1)
    t1_df["total_ask_depth"] = np.sum(ask_v_matrix, axis=1)
    t1_df["total_book_depth"] = t1_df["total_bid_depth"] + t1_df["total_ask_depth"]
    t1_df["depth_ratio_l1_to_topK"] = (bid_v1 + ask_v1) / (t1_df["total_book_depth"] + 1e-6)
    
    return t1_df, list(t1_df.columns)
