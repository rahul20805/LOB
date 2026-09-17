"""
Market microstructure analytics and statistics for LOB data.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


def compute_microstructure_stats(
    df: pd.DataFrame,
    spread_col: str = "spread",
    mid_col: str = "mid_price",
    vol_imbalance_col: str = "volume_imbalance"
) -> Dict[str, float]:
    """
    Computes key market microstructure summary statistics.
    """
    stats = {}
    if spread_col in df.columns:
        stats["mean_spread"] = float(df[spread_col].mean())
        stats["median_spread"] = float(df[spread_col].median())
        stats["std_spread"] = float(df[spread_col].std())
        stats["spread_to_mid_bps"] = float((df[spread_col] / df[mid_col]).mean() * 10000)

    if vol_imbalance_col in df.columns:
        stats["mean_imbalance"] = float(df[vol_imbalance_col].mean())
        stats["imbalance_autocorr_lag1"] = float(df[vol_imbalance_col].autocorr(lag=1))

    if mid_col in df.columns:
        returns = df[mid_col].pct_change().dropna()
        stats["tick_volatility"] = float(returns.std())
        stats["return_autocorr_lag1"] = float(returns.autocorr(lag=1))
        stats["kurtosis"] = float(returns.kurtosis())
        stats["skewness"] = float(returns.skew())

    return stats
