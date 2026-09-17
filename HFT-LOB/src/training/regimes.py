"""
Regime-Adaptive Protocol and Identification Engine (Section 7.1 & Eq 29).
Classifies market states into 4 quadrants:
- HV-T: High-Vol Trending (sigma > sigma_bar and rho > rho_bar)
- HV-M: High-Vol Mean-Reverting (sigma > sigma_bar and rho <= rho_bar)
- LV-T: Low-Vol Trending (sigma <= sigma_bar and rho > rho_bar)
- LV-M: Low-Vol Mean-Reverting (sigma <= sigma_bar and rho <= rho_bar)
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd


class MarketRegimeDetector:
    """
    Identifies microstructure market regimes via trailing volatility and return autocorrelation.
    """
    def __init__(self, window_len: int = 30):
        self.window_len = window_len
        self.sigma_bar: float = 0.0
        self.rho_bar: float = 0.0

    def fit_reference_medians(self, returns: np.ndarray):
        """
        Calibrates sigma_bar and rho_bar medians on training fold.
        """
        ret_series = pd.Series(returns)
        rolling_vol = ret_series.rolling(self.window_len, min_periods=5).std().dropna()
        
        # Rolling lag-1 autocorrelation
        rolling_autocorr = ret_series.rolling(self.window_len, min_periods=5).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else 0.0, raw=False
        ).dropna()
        
        self.sigma_bar = float(rolling_vol.median()) if len(rolling_vol) > 0 else 0.001
        self.rho_bar = float(rolling_autocorr.median()) if len(rolling_autocorr) > 0 else 0.0

    def classify_regimes(self, returns: np.ndarray) -> np.ndarray:
        """
        Equation (29): Classifies each time step into HV-T, HV-M, LV-T, LV-M.
        """
        ret_series = pd.Series(returns)
        rolling_vol = ret_series.rolling(self.window_len, min_periods=1).std().fillna(self.sigma_bar).values
        rolling_autocorr = ret_series.rolling(self.window_len, min_periods=5).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 3 else self.rho_bar, raw=False
        ).fillna(self.rho_bar).values
        
        regimes = []
        for sig, rho in zip(rolling_vol, rolling_autocorr):
            if sig > self.sigma_bar:
                if rho > self.rho_bar:
                    regimes.append("HV-T")
                else:
                    regimes.append("HV-M")
            else:
                if rho > self.rho_bar:
                    regimes.append("LV-T")
                else:
                    regimes.append("LV-M")
                    
        return np.array(regimes)
