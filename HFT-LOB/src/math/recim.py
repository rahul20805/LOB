"""
Realistic Execution Cost Integration Model (RECIM)
Implements Section 6 & Algorithm 1 of Chaudhary & Kushwaha (2024).
Provides rigorous execution cost accounting: half-spread, exchange fees,
crossover market impact, and limit order fill survival modeling.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.math.formulas import compute_market_impact


class RECIMCostCalculator:
    """
    Unified transaction cost calculator and execution engine for high-frequency trading.
    """
    def __init__(
        self,
        fee_rate: float = 0.0002,       # 2 bps exchange fee
        eta: float = 0.5,               # Market impact liquidity coefficient
        xi_star: float = 1e-3,          # Linear-to-square-root crossover threshold
        base_hazard_rate: float = 0.15, # Fill hazard rate for passive limit orders
        slippage_std: float = 0.00005   # Random execution jitter
    ):
        self.fee_rate = fee_rate
        self.eta = eta
        self.xi_star = xi_star
        self.base_hazard_rate = base_hazard_rate
        self.slippage_std = slippage_std

    def compute_transaction_cost(
        self,
        order_size: float,
        best_ask: float,
        best_bid: float,
        realized_vol: float,
        daily_volume: float,
        order_type: str = "market",
        horizon: int = 10,
        depth_level: int = 1
    ) -> Dict[str, float]:
        """
        Implements Algorithm 1 & Eq. (24)-(28).
        """
        spread = max(best_ask - best_bid, 1e-6)
        half_spread = spread / 2.0
        mid_price = 0.5 * (best_ask + best_bid)
        
        # 1. Exchange fee
        fee_cost = self.fee_rate * mid_price * abs(order_size)
        
        # 2. Market impact
        impact_cost = compute_market_impact(
            order_size=order_size,
            daily_volume=daily_volume,
            realized_vol=realized_vol,
            eta=self.eta,
            xi_star=self.xi_star
        ) * mid_price * abs(order_size)
        
        # 3. Execution type handling
        if order_type.lower() == "market":
            spread_cost = half_spread * abs(order_size)
            total_tc = spread_cost + fee_cost + impact_cost
            p_fill = 1.0
            is_filled = True
        else:
            # Limit order: Equation (27) fill probability
            # lambda(u) estimated hazard rate decays with deeper levels
            level_hazard = self.base_hazard_rate / (depth_level ** 0.5)
            # Survival function integration over horizon
            p_fill = 1.0 - np.exp(-level_hazard * horizon)
            
            # Stochastic execution simulation
            is_filled = np.random.rand() < p_fill
            if is_filled:
                # Passive fill earns the half spread, pays maker fee, 0 impact
                total_tc = fee_cost
            else:
                # Missed fill: fallback to market order + opportunity cost
                spread_cost = half_spread * abs(order_size)
                total_tc = spread_cost + fee_cost + impact_cost

        return {
            "total_tc": float(total_tc),
            "half_spread_cost": float(half_spread * abs(order_size)),
            "fee_cost": float(fee_cost),
            "impact_cost": float(impact_cost),
            "p_fill": float(p_fill),
            "is_filled": bool(is_filled)
        }

    def compute_net_pnl(
        self,
        signal: int,
        order_size: float,
        entry_mid: float,
        exit_mid: float,
        entry_ask: float,
        entry_bid: float,
        realized_vol: float,
        daily_volume: float,
        order_type: str = "market",
        horizon: int = 10
    ) -> Dict[str, float]:
        """
        Compute net P&L contribution pi_t (Eq. 23 & Algorithm 1 Line 12).
        pi_t = d_t * Q * (P_exec_{t+Delta} - P_exec_t) - TC
        """
        if signal == 0 or order_size == 0:
            return {
                "gross_pnl": 0.0,
                "total_tc": 0.0,
                "net_pnl": 0.0,
                "gross_return": 0.0,
                "net_return": 0.0
            }

        # Execution pricing
        if signal > 0: # Buy
            entry_exec = entry_ask if order_type == "market" else entry_bid
        else: # Sell
            entry_exec = entry_bid if order_type == "market" else entry_ask

        # Compute transaction cost
        cost_dict = self.compute_transaction_cost(
            order_size=order_size,
            best_ask=entry_ask,
            best_bid=entry_bid,
            realized_vol=realized_vol,
            daily_volume=daily_volume,
            order_type=order_type,
            horizon=horizon
        )
        tc = cost_dict["total_tc"]

        # Gross price change
        price_diff = exit_mid - entry_mid
        gross_pnl = float(signal * order_size * price_diff)
        net_pnl = float(gross_pnl - tc)

        # Returns relative to entry position capital
        capital = entry_mid * abs(order_size)
        gross_return = float(gross_pnl / capital) if capital > 0 else 0.0
        net_return = float(net_pnl / capital) if capital > 0 else 0.0

        return {
            "gross_pnl": gross_pnl,
            "total_tc": tc,
            "net_pnl": net_pnl,
            "gross_return": gross_return,
            "net_return": net_return,
            "half_spread_cost": cost_dict["half_spread_cost"],
            "fee_cost": cost_dict["fee_cost"],
            "impact_cost": cost_dict["impact_cost"],
            "p_fill": cost_dict["p_fill"]
        }
