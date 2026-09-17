"""
Mathematical formulas and microstructure computations for LOB-PROFIT Framework.
Implements Equations (1) to (14), (25)-(26), (29)-(30) with numerical safeguards.
"""

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd


def compute_spread(ask_price: np.ndarray, bid_price: np.ndarray) -> np.ndarray:
    """
    Equation (8): Compute bid-ask spread s_t = P_1,t^a - P_1,t^b > 0.
    """
    spread = ask_price - bid_price
    return np.maximum(spread, 1e-8)


def compute_mid_price(ask_price: np.ndarray, bid_price: np.ndarray) -> np.ndarray:
    """
    Equation (2) & (9): Compute mid-price m_t = (P_1,t^a + P_1,t^b) / 2.
    """
    return 0.5 * (ask_price + bid_price)


def compute_volume_imbalance(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> np.ndarray:
    """
    Equation (10): Compute multi-level volume imbalance psi_t.
    psi_t = (sum(V_k^b) - sum(V_k^a)) / (sum(V_k^b) + sum(V_k^a))
    """
    sum_bid = np.sum(bid_volumes, axis=-1)
    sum_ask = np.sum(ask_volumes, axis=-1)
    total_vol = sum_bid + sum_ask
    safe_denom = np.where(total_vol == 0, 1.0, total_vol)
    return (sum_bid - sum_ask) / safe_denom


def compute_weighted_mid_price(best_bid_p: np.ndarray, best_bid_v: np.ndarray,
                               best_ask_p: np.ndarray, best_ask_v: np.ndarray) -> np.ndarray:
    """
    Equation (11): Compute volume-weighted mid-price (microprice) m_t^w.
    m_t^w = (P_1^b * V_1^a + P_1^a * V_1^b) / (V_1^a + V_1^b)
    """
    total_v = best_ask_v + best_bid_v
    safe_denom = np.where(total_v == 0, 1.0, total_v)
    return (best_bid_p * best_ask_v + best_ask_p * best_bid_v) / safe_denom


def compute_smoothed_return(mid_prices: np.ndarray, horizon: int = 10,
                            smoothing_window: int = 5) -> np.ndarray:
    """
    Equation (3): Compute forward smoothed relative return l_t.
    l_t = (m_bar_{t+Delta} - m_bar_t) / m_bar_t
    where m_bar_t is rolling mean over H past observations.
    """
    mid_series = pd.Series(mid_prices)
    m_bar = mid_series.rolling(window=smoothing_window, min_periods=1).mean().values
    
    # Forward shifted smoothed mid-price
    m_bar_forward = np.roll(m_bar, -horizon)
    m_bar_forward[-horizon:] = np.nan
    
    safe_denom = np.where(m_bar == 0, 1e-6, m_bar)
    l_t = (m_bar_forward - m_bar) / safe_denom
    return l_t


def assign_ternary_labels(relative_returns: np.ndarray, threshold: float = 0.0002) -> np.ndarray:
    """
    Equation (4): Assign ternary direction label y_t in {-1, 0, +1}.
    y_t = +1 if l_t > alpha
          -1 if l_t < -alpha
           0 otherwise
    """
    labels = np.zeros_like(relative_returns, dtype=np.int64)
    labels[relative_returns > threshold] = 1
    labels[relative_returns < -threshold] = -1
    # Preserve NaNs as -999 or handle appropriately
    labels[np.isnan(relative_returns)] = 0
    return labels


def compute_level_ofi(bid_p_curr: np.ndarray, bid_p_prev: np.ndarray,
                      bid_v_curr: np.ndarray, bid_v_prev: np.ndarray,
                      ask_p_curr: np.ndarray, ask_p_prev: np.ndarray,
                      ask_v_curr: np.ndarray, ask_v_prev: np.ndarray) -> np.ndarray:
    """
    Equations (5), (6), (7): Order Flow Imbalance for level k.
    Delta W_k,t^b = V_k,t^b * 1{P_k,t^b >= P_k,t-1^b} - V_k,t-1^b * 1{P_k,t^b <= P_k,t-1^b}
    Delta W_k,t^a = V_k,t^a * 1{P_k,t^a <= P_k,t-1^a} - V_k,t-1^a * 1{P_k,t^a >= P_k,t-1^a}
    OFI_t^(k) = Delta W_k,t^b - Delta W_k,t^a
    """
    # Bid side update
    delta_w_b = np.zeros_like(bid_v_curr, dtype=np.float64)
    bid_p_gt = bid_p_curr > bid_p_prev
    bid_p_eq = bid_p_curr == bid_p_prev
    bid_p_lt = bid_p_curr < bid_p_prev
    
    delta_w_b[bid_p_gt] = bid_v_curr[bid_p_gt]
    delta_w_b[bid_p_eq] = bid_v_curr[bid_p_eq] - bid_v_prev[bid_p_eq]
    delta_w_b[bid_p_lt] = -bid_v_prev[bid_p_lt]
    
    # Ask side update
    delta_w_a = np.zeros_like(ask_v_curr, dtype=np.float64)
    ask_p_lt = ask_p_curr < ask_p_prev
    ask_p_eq = ask_p_curr == ask_p_prev
    ask_p_gt = ask_p_curr > ask_p_prev
    
    delta_w_a[ask_p_lt] = ask_v_curr[ask_p_lt]
    delta_w_a[ask_p_eq] = ask_v_curr[ask_p_eq] - ask_v_prev[ask_p_eq]
    delta_w_a[ask_p_gt] = -ask_v_prev[ask_p_gt]
    
    return delta_w_b - delta_w_a


def compute_market_impact(order_size: float, daily_volume: float,
                          realized_vol: float, eta: float = 0.5,
                          xi_star: float = 1e-3) -> float:
    """
    Equations (25), (26): Bucci et al. crossover market impact.
    I(|Q|, t) = sigma_t * f(|Q| / V_t^daily)
    where f(xi) = eta * xi if xi <= xi_star else eta * sqrt(xi_star) * sqrt(xi).
    """
    if daily_volume <= 0:
        daily_volume = 1e6
    xi = abs(order_size) / daily_volume
    if xi <= xi_star:
        f_xi = eta * xi
    else:
        f_xi = eta * np.sqrt(xi_star) * np.sqrt(xi)
    return float(realized_vol * f_xi)


def compute_pdi(metric_train: float, metric_test: float, eps: float = 1e-6) -> float:
    """
    Equation (30): Performance Degradation Index (PDI).
    PDI(M) = max(0, M(R_train) - M(R_test)) / (|M(R_train)| + eps)
    """
    deg = max(0.0, metric_train - metric_test)
    denom = abs(metric_train) + eps
    return float(deg / denom)
