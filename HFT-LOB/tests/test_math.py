"""
Unit tests for mathematical formulas and RECIM cost engine.
"""

import numpy as np
import pytest
from src.math.formulas import (
    compute_spread,
    compute_mid_price,
    compute_volume_imbalance,
    compute_weighted_mid_price,
    compute_smoothed_return,
    assign_ternary_labels,
    compute_level_ofi,
    compute_market_impact,
    compute_pdi
)
from src.math.recim import RECIMCostCalculator


def test_spread_and_mid():
    ask = np.array([100.5, 101.0, 102.0])
    bid = np.array([100.0, 100.5, 101.5])
    
    spread = compute_spread(ask, bid)
    mid = compute_mid_price(ask, bid)
    
    np.testing.assert_allclose(spread, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(mid, [100.25, 100.75, 101.75])


def test_volume_imbalance():
    bid_v = np.array([[10, 20], [30, 40]])
    ask_v = np.array([[10, 20], [10, 20]])
    
    psi = compute_volume_imbalance(bid_v, ask_v)
    assert psi[0] == 0.0 # Equal bid/ask
    assert psi[1] == (70 - 30) / (70 + 30) # 40 / 100 = 0.4


def test_weighted_mid_price():
    best_bid_p = np.array([100.0])
    best_bid_v = np.array([10.0])
    best_ask_p = np.array([101.0])
    best_ask_v = np.array([30.0])
    
    # m^w = (100 * 30 + 101 * 10) / (30 + 10) = 4010 / 40 = 100.25
    m_w = compute_weighted_mid_price(best_bid_p, best_bid_v, best_ask_p, best_ask_v)
    np.testing.assert_allclose(m_w, [100.25])


def test_ofi_calculation():
    # Price rises: current bid > prev bid -> delta_w_b = curr_v
    bid_p_curr = np.array([100.5])
    bid_p_prev = np.array([100.0])
    bid_v_curr = np.array([15.0])
    bid_v_prev = np.array([10.0])
    
    # Ask price rises: current ask > prev ask -> delta_w_a = -prev_v
    ask_p_curr = np.array([101.5])
    ask_p_prev = np.array([101.0])
    ask_v_curr = np.array([20.0])
    ask_v_prev = np.array([25.0])
    
    # OFI = 15.0 - (-25.0) = 40.0
    ofi = compute_level_ofi(
        bid_p_curr, bid_p_prev, bid_v_curr, bid_v_prev,
        ask_p_curr, ask_p_prev, ask_v_curr, ask_v_prev
    )
    np.testing.assert_allclose(ofi, [40.0])


def test_market_impact_crossover():
    # Linear regime (xi <= 0.001)
    imp_linear = compute_market_impact(
        order_size=10, daily_volume=100000, realized_vol=0.01, eta=0.5, xi_star=0.001
    )
    # xi = 10 / 100000 = 0.0001 <= 0.001 -> f(xi) = 0.5 * 0.0001 = 0.00005 -> imp = 0.01 * 0.00005 = 5e-7
    np.testing.assert_allclose(imp_linear, 5e-7)
    
    # Square root regime (xi > 0.001)
    imp_sqrt = compute_market_impact(
        order_size=1000, daily_volume=100000, realized_vol=0.01, eta=0.5, xi_star=0.001
    )
    assert imp_sqrt > imp_linear


def test_pdi_calculation():
    # Train Sharpe 2.0, Test Sharpe 1.0 -> PDI = (2.0 - 1.0) / 2.0 = 0.5
    pdi = compute_pdi(2.0, 1.0)
    assert abs(pdi - 0.5) < 1e-4
    
    # Train Sharpe 1.0, Test Sharpe 1.5 -> degradation is 0 -> PDI = 0
    pdi_gain = compute_pdi(1.0, 1.5)
    assert pdi_gain == 0.0


def test_recim_calculator():
    recim = RECIMCostCalculator(fee_rate=0.0002, eta=0.5)
    cost = recim.compute_transaction_cost(
        order_size=100,
        best_ask=100.1,
        best_bid=100.0,
        realized_vol=0.01,
        daily_volume=100000,
        order_type="market"
    )
    assert cost["total_tc"] > 0
    assert cost["half_spread_cost"] > 0
    assert cost["fee_cost"] > 0

    pnl_res = recim.compute_net_pnl(
        signal=1,
        order_size=100,
        entry_mid=100.05,
        exit_mid=100.50,
        entry_ask=100.1,
        entry_bid=100.0,
        realized_vol=0.01,
        daily_volume=100000,
        order_type="market"
    )
    assert pnl_res["gross_pnl"] > 0
    assert pnl_res["net_pnl"] < pnl_res["gross_pnl"] # Costs deducted
