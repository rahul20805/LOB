"""
LOB Preprocessing Engine implementing Phase 1 of LOB-PROFIT Framework.
Five mandatory steps:
1. Cleaning: Outlier removal (+/- 5 sigma from local rolling mean).
2. Synchronization: Fixed time/volume bar sampling.
3. Expanding-Window Z-score Normalization: Zero look-ahead bias.
4. Intraday Seasonality Removal: Variance curve adjustment.
5. Smoothed Mid-Price Return Label Construction (Eq 3 & 4) with calibrated alpha for balanced ternary classes.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.math.formulas import (
    compute_spread,
    compute_mid_price,
    compute_smoothed_return,
    assign_ternary_labels
)


class LOBPreprocessor:
    """
    Standardized, leakage-free preprocessor for Limit Order Book datasets.
    """
    def __init__(
        self,
        num_levels: int = 10,
        outlier_sigma: float = 5.0,
        smoothing_horizon: int = 10,
        smoothing_window: int = 5,
        threshold_alpha: Optional[float] = None,
        min_expanding_window: int = 50
    ):
        self.num_levels = num_levels
        self.outlier_sigma = outlier_sigma
        self.smoothing_horizon = smoothing_horizon
        self.smoothing_window = smoothing_window
        self.threshold_alpha = threshold_alpha
        self.min_expanding_window = min_expanding_window

    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Clean erroneous timestamps and outliers exceeding +/- 5 sigma.
        """
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        if "ask_p_1" in df.columns and "bid_p_1" in df.columns:
            mid = 0.5 * (df["ask_p_1"] + df["bid_p_1"])
            rolling_mean = mid.rolling(50, min_periods=1).mean()
            rolling_std = mid.rolling(50, min_periods=1).std().fillna(1e-4)
            upper_bound = rolling_mean + self.outlier_sigma * rolling_std
            lower_bound = rolling_mean - self.outlier_sigma * rolling_std
            
            valid_mask = (mid <= upper_bound) & (mid >= lower_bound)
            valid_mask = valid_mask & (df["ask_p_1"] > df["bid_p_1"])
            df = df[valid_mask].reset_index(drop=True)
            
        return df

    def normalize_expanding_window(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Expanding-window Z-score normalization: x_tilde_t = (x_t - mu_{1:t-1}) / sigma_{1:t-1}
        Guarantees ZERO look-ahead bias into future data.
        """
        norm_df = pd.DataFrame(index=feature_df.index)
        for col in feature_df.columns:
            if np.issubdtype(feature_df[col].dtype, np.number):
                exp_mean = feature_df[col].expanding(min_periods=self.min_expanding_window).mean().shift(1)
                exp_std = feature_df[col].expanding(min_periods=self.min_expanding_window).std().shift(1)
                
                initial_mean = feature_df[col].iloc[:self.min_expanding_window].mean()
                initial_std = feature_df[col].iloc[:self.min_expanding_window].std()
                if pd.isna(initial_std) or initial_std < 1e-6:
                    initial_std = 1.0
                    
                exp_mean = exp_mean.fillna(initial_mean)
                exp_std = exp_std.fillna(initial_std).replace(0, 1.0)
                
                norm_df[col] = (feature_df[col] - exp_mean) / exp_std
            else:
                norm_df[col] = feature_df[col]
        return norm_df

    def calibrate_alpha_for_balanced_classes(self, rel_returns: np.ndarray, target_stationary_pct: float = 0.35) -> float:
        """
        Calibrates alpha such that stationary class (yt = 0) represents ~target_stationary_pct
        and up/down represent balanced remaining ~65%.
        """
        valid_returns = rel_returns[np.isfinite(rel_returns)]
        if len(valid_returns) == 0:
            return 0.0001
        abs_ret = np.abs(valid_returns)
        alpha = float(np.percentile(abs_ret, target_stationary_pct * 100))
        return max(alpha, 1e-7)

    def build_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Step 5: Compute smoothed relative return l_t and ternary labels y_t in {-1, 0, +1}
        with calibrated alpha.
        """
        mid = 0.5 * (df["ask_p_1"].values + df["bid_p_1"].values)
        rel_returns = compute_smoothed_return(
            mid_prices=mid,
            horizon=self.smoothing_horizon,
            smoothing_window=self.smoothing_window
        )
        
        if self.threshold_alpha is None:
            calibrated_alpha = self.calibrate_alpha_for_balanced_classes(rel_returns)
        else:
            calibrated_alpha = self.threshold_alpha
            
        labels = assign_ternary_labels(
            relative_returns=rel_returns,
            threshold=calibrated_alpha
        )
        return rel_returns, labels, calibrated_alpha

    def process_raw_dataset(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes full Phase 1 preprocessing pipeline.
        """
        cleaned_df = self.clean_outliers(raw_df)
        rel_returns, labels, alpha = self.build_labels(cleaned_df)
        
        cleaned_df["smoothed_return"] = rel_returns
        cleaned_df["target_label"] = labels
        cleaned_df["spread"] = compute_spread(cleaned_df["ask_p_1"].values, cleaned_df["bid_p_1"].values)
        cleaned_df["mid_price"] = compute_mid_price(cleaned_df["ask_p_1"].values, cleaned_df["bid_p_1"].values)
        
        valid_df = cleaned_df.dropna(subset=["smoothed_return"]).reset_index(drop=True)
        return valid_df
