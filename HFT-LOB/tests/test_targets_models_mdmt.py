"""
Unit tests for target construction, models, MDMT metrics, and RAEP regimes.
"""

import numpy as np
import pytest
import torch
from src.targets.target_builder import TargetBuilder
from src.models.baselines import BaselineModelZoo
from src.models.deep_learning import DeepLOBSpatialCNN, DeepLOBSpatiotemporal, LiTTransformer
from src.models.ensembles import DynamicSharpeEnsemble
from src.evaluation.mdmt_metrics import MDMTEvaluator
from src.evaluation.statistical_tests import mcnemar_test, wilcoxon_signed_rank_test, compute_bootstrap_ci
from src.training.regimes import MarketRegimeDetector


def test_target_builder():
    tb = TargetBuilder(horizon=5, smoothing_window=3, threshold_alpha=0.001)
    mids = np.linspace(100, 110, 50)
    rel_ret, tern_y, cls_y = tb.build_targets(mids)
    
    assert len(rel_ret) == 50
    assert len(tern_y) == 50
    assert len(cls_y) == 50
    # Price is consistently rising, valid forward returns should be positive (class 2)
    assert cls_y[10] == 2


def test_baseline_models():
    X = np.random.randn(60, 20)
    y = np.random.choice([0, 1, 2], size=60)
    
    for m_name in ["logistic", "rf", "xgb"]:
        model = BaselineModelZoo.create_model(m_name)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (60, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(60), rtol=1e-5)


def test_deep_models_forward():
    batch_size = 4
    seq_len = 15
    in_feat = 40
    dummy_x = torch.randn(batch_size, seq_len, in_feat)
    
    cnn = DeepLOBSpatialCNN(in_features=in_feat, seq_len=seq_len, num_classes=3)
    out_cnn = cnn(dummy_x)
    assert out_cnn.shape == (batch_size, 3)
    
    spatio = DeepLOBSpatiotemporal(in_features=in_feat, seq_len=seq_len, num_classes=3, lstm_hidden=32)
    out_spatio = spatio(dummy_x)
    assert out_spatio.shape == (batch_size, 3)
    
    lit = LiTTransformer(in_features=in_feat, seq_len=seq_len, d_model=32, nhead=2, num_layers=1, num_classes=3)
    out_lit = lit(dummy_x)
    assert out_lit.shape == (batch_size, 3)


def test_dynamic_sharpe_ensemble():
    ens = DynamicSharpeEnsemble(model_names=["m1", "m2"])
    ens.update_performance({"m1": 0.01, "m2": -0.01})
    
    p1 = np.array([[0.1, 0.2, 0.7]])
    p2 = np.array([[0.4, 0.4, 0.2]])
    comb = ens.predict_proba({"m1": p1, "m2": p2})
    assert comb.shape == (1, 3)
    assert ens.weights["m1"] > ens.weights["m2"]


def test_mdmt_evaluation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    
    tier_a = MDMTEvaluator.evaluate_tier_a_statistical(y_true, y_pred)
    assert tier_a["accuracy"] == 5 / 6
    assert "weighted_f1" in tier_a
    assert "cohen_kappa" in tier_a
    
    signals = np.array([1, -1, 1, 0, 0, 1])
    act_ret = np.array([0.01, -0.01, -0.005, 0.0, 0.0, 0.02])
    net_ret = np.array([0.008, 0.007, -0.006, 0.0, 0.0, 0.015])
    
    tier_b = MDMTEvaluator.evaluate_tier_b_signal_quality(signals, act_ret, net_ret)
    assert tier_b["directional_accuracy"] > 0
    assert tier_b["transaction_hit_rate"] == 3 / 4 # 3 out of 4 active trades net > 0
    
    tier_c = MDMTEvaluator.evaluate_tier_c_trade_level(net_ret, signals)
    assert tier_c["hit_rate"] == 0.75
    assert tier_c["profit_factor"] > 1.0
    
    net_pnls = np.array([10.0, 15.0, -5.0, 20.0, -2.0, 18.0])
    tier_d = MDMTEvaluator.evaluate_tier_d_portfolio_level(net_pnls)
    assert tier_d["cumulative_pnl"] == 56.0
    assert tier_d["annualized_sharpe"] > 0


def test_regime_detector():
    detector = MarketRegimeDetector(window_len=10)
    returns = np.sin(np.linspace(0, 20, 100)) * 0.01
    detector.fit_reference_medians(returns)
    regimes = detector.classify_regimes(returns)
    assert len(regimes) == 100
    assert set(regimes).issubset({"HV-T", "HV-M", "LV-T", "LV-M"})
