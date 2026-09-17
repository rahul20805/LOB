"""
Tier 3: Time-Sensitive Dynamic Features.
Implements Section 4.2.4:
- Price momentum rho_t^(ell) = m_t - m_{t-ell} for ell in {1, 5, 10, 30}
- Spread dynamics Delta s_t = s_t - s_{t-1}
- Realized volatility sigma_hat_t = sqrt(sum (m_tau - m_{tau-1})^2)
- Time-of-day sinusoidal encodings
- Queue dynamics Delta V_{1,t}^b, Delta V_{1,t}^a
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


def extract_tier3_dynamics(
    df: pd.DataFrame,
    momentum_lags: List[int] = [1, 5, 10, 30],
    vol_windows: List[int] = [10, 30, 50]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts Tier 3 dynamic and time-sensitive microstructure features.
    """
    t3_df = pd.DataFrame(index=df.index)
    
    # 1. Mid-price series
    if "mid_price" in df.columns:
        mid_series = df["mid_price"]
    else:
        mid_series = 0.5 * (df["ask_p_1"] + df["bid_p_1"])
        
    # 2. Multi-horizon Price Momentum rho^(ell)
    for lag in momentum_lags:
        t3_df[f"price_momentum_lag_{lag}"] = (mid_series - mid_series.shift(lag)).fillna(0)
        t3_df[f"pct_return_lag_{lag}"] = mid_series.pct_change(lag).fillna(0)

    # 3. Spread dynamics Delta s_t
    if "spread" in df.columns:
        spread_series = df["spread"]
    else:
        spread_series = df["ask_p_1"] - df["bid_p_1"]
    t3_df["spread_change_lag1"] = spread_series.diff(1).fillna(0)
    t3_df["spread_rolling_mean_10"] = spread_series.rolling(10, min_periods=1).mean()

    # 4. Realized Volatility sigma_hat_t
    ret_diff = mid_series.diff(1).fillna(0)
    for w in vol_windows:
        t3_df[f"realized_vol_w{w}"] = np.sqrt((ret_diff ** 2).rolling(window=w, min_periods=1).sum())

    # 5. Queue dynamics Delta V_1^b, Delta V_1^a
    if "bid_v_1" in df.columns and "ask_v_1" in df.columns:
        t3_df["queue_change_bid_l1"] = df["bid_v_1"].diff(1).fillna(0)
        t3_df["queue_change_ask_l1"] = df["ask_v_1"].diff(1).fillna(0)
        t3_df["queue_imbalance_diff"] = t3_df["queue_change_bid_l1"] - t3_df["queue_change_ask_l1"]

    # 6. Time-of-day Sinusoidal Encoding
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        if dt.notna().all():
            sec_of_day = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
        else:
            sec_of_day = (np.arange(len(df)) * 0.05) % 86400
    else:
        sec_of_day = (np.arange(len(df)) * 0.05) % 86400
        
    t3_df["time_sin"] = np.sin(2 * np.pi * sec_of_day / 86400.0)
    t3_df["time_cos"] = np.cos(2 * np.pi * sec_of_day / 86400.0)

    return t3_df, list(t3_df.columns)
