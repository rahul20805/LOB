"""
Tier 0: Raw Limit Order Book State Features.
Constructs 2D LOB snapshot matrix X_t in R^{K x 4}
Columns: (P_k^b, V_k^b, P_k^a, V_k^a) for k = 1 ... K.
Standard input for CNN-based architectures (DeepLOB).
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


def extract_tier0_raw_lob(df: pd.DataFrame, num_levels: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts raw 40-dimensional LOB state vector or 2D image matrix.
    """
    feature_names = []
    cols = []
    for k in range(1, num_levels + 1):
        for side, prefix in [("bid", "bid"), ("ask", "ask")]:
            p_col = f"{prefix}_p_{k}"
            v_col = f"{prefix}_v_{k}"
            if p_col in df.columns and v_col in df.columns:
                cols.extend([p_col, v_col])
                feature_names.extend([p_col, v_col])
                
    raw_matrix = df[cols].values
    return raw_matrix, feature_names
