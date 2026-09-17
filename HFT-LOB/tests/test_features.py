"""
Unit tests for 4-tier feature engineering taxonomy.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.tier0_raw import extract_tier0_raw_lob
from src.features.tier1_microstructure import extract_tier1_microstructure
from src.features.tier2_ofi import extract_tier2_ofi
from src.features.tier3_dynamics import extract_tier3_dynamics
from src.features.feature_engine import FeatureEngine


@pytest.fixture
def sample_lob_df():
    np.random.seed(42)
    n = 100
    data = {"timestamp": np.arange(n) * 0.05}
    base_price = 100.0
    for k in range(1, 11):
        data[f"bid_p_{k}"] = base_price - 0.05 * k + np.random.normal(0, 0.01, n)
        data[f"bid_v_{k}"] = np.random.exponential(10.0, n) + 1.0
        data[f"ask_p_{k}"] = base_price + 0.05 * k + np.random.normal(0, 0.01, n)
        data[f"ask_v_{k}"] = np.random.exponential(10.0, n) + 1.0
    return pd.DataFrame(data)


def test_tier0_extraction(sample_lob_df):
    raw_mat, feat_names = extract_tier0_raw_lob(sample_lob_df, num_levels=10)
    assert raw_mat.shape == (100, 40)
    assert len(feat_names) == 40
    assert "bid_p_1" in feat_names
    assert "ask_v_10" in feat_names


def test_tier1_extraction(sample_lob_df):
    t1_df, names = extract_tier1_microstructure(sample_lob_df, num_levels=10)
    assert "spread" in t1_df.columns
    assert "mid_price" in t1_df.columns
    assert "microprice" in t1_df.columns
    assert "volume_imbalance_topK" in t1_df.columns
    assert (t1_df["spread"] > 0).all()


def test_tier2_ofi_extraction(sample_lob_df):
    t2_df, names, pca_vec = extract_tier2_ofi(sample_lob_df, num_levels=10)
    assert "ofi_integrated" in t2_df.columns
    assert "cii_window_5" in t2_df.columns
    assert len(pca_vec) == 10


def test_tier3_dynamics_extraction(sample_lob_df):
    t3_df, names = extract_tier3_dynamics(sample_lob_df)
    assert "price_momentum_lag_1" in t3_df.columns
    assert "spread_change_lag1" in t3_df.columns
    assert "realized_vol_w10" in t3_df.columns
    assert "time_sin" in t3_df.columns


def test_unified_feature_engine(sample_lob_df):
    engine = FeatureEngine(num_levels=10, sequence_length=15)
    feat_df, tier_map = engine.extract_all_features(sample_lob_df, is_training=True)
    assert feat_df.shape[0] == 100
    assert feat_df.shape[1] > 50
    assert "Tier 0 (Raw LOB)" in tier_map
    assert "Tier 2 (OFI)" in tier_map

    X_seq, y_seq = engine.build_sequence_tensors(feat_df.values, np.zeros(100, dtype=np.int64), seq_len=15)
    assert X_seq.shape == (86, 15, feat_df.shape[1])
    assert len(y_seq) == 86
