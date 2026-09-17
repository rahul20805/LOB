"""
Event-Driven Trading Simulator implementing Section 8.2 & RECIM Algorithm 1.
Computes net-of-cost trade execution, P&L, Transaction Hit Rate, and predictions log.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.math.recim import RECIMCostCalculator
from src.trading.risk_manager import RiskManager


class EventDrivenSimulator:
    """
    Simulates high-frequency trade execution under realistic market microstructure friction.
    """
    def __init__(
        self,
        recim_calculator: Optional[RECIMCostCalculator] = None,
        risk_manager: Optional[RiskManager] = None,
        default_order_size: float = 1.0,
        daily_volume: float = 1000000.0,
        confidence_threshold: float = 0.45
    ):
        self.recim = recim_calculator if recim_calculator is not None else RECIMCostCalculator()
        self.risk = risk_manager if risk_manager is not None else RiskManager()
        self.default_order_size = default_order_size
        self.daily_volume = daily_volume
        self.confidence_threshold = confidence_threshold

    def run_simulation(
        self,
        df_lob: pd.DataFrame,
        predicted_probas: np.ndarray,
        horizon: int = 10,
        order_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Executes event-driven simulation over test LOB snapshots.
        """
        self.risk.reset()
        num_events = len(predicted_probas)
        
        trade_records = []
        signals = np.zeros(num_events, dtype=np.int64)
        net_pnls = np.zeros(num_events, dtype=np.float64)
        gross_pnls = np.zeros(num_events, dtype=np.float64)
        total_costs = np.zeros(num_events, dtype=np.float64)
        net_returns = np.zeros(num_events, dtype=np.float64)
        actual_labels = []
        
        # Realized volatility estimate
        mid_series = 0.5 * (df_lob["ask_p_1"].values + df_lob["bid_p_1"].values)
        ret_diff = pd.Series(mid_series).pct_change().fillna(0).values
        rolling_vol = pd.Series(ret_diff).rolling(30, min_periods=1).std().fillna(0.001).values

        for t in range(num_events):
            if t + horizon >= len(df_lob):
                break
                
            probs = predicted_probas[t]
            p_down, p_stat, p_up = probs[0], probs[1], probs[2]
            
            # Phase 4 Signal conversion (Eq 15)
            dt = 0
            if p_up > self.confidence_threshold and p_up == max(probs):
                dt = 1
            elif p_down > self.confidence_threshold and p_down == max(probs):
                dt = -1
                
            signals[t] = dt
            
            if dt != 0:
                allowed, exec_size, reason = self.risk.check_trade_allowed(dt, self.default_order_size)
                if allowed and exec_size > 0:
                    entry_ask = float(df_lob["ask_p_1"].iloc[t])
                    entry_bid = float(df_lob["bid_p_1"].iloc[t])
                    entry_mid = float(mid_series[t])
                    exit_mid = float(mid_series[t + horizon])
                    sig_vol = float(rolling_vol[t])
                    
                    pnl_res = self.recim.compute_net_pnl(
                        signal=dt,
                        order_size=exec_size,
                        entry_mid=entry_mid,
                        exit_mid=exit_mid,
                        entry_ask=entry_ask,
                        entry_bid=entry_bid,
                        realized_vol=sig_vol,
                        daily_volume=self.daily_volume,
                        order_type=order_type,
                        horizon=horizon
                    )
                    
                    net_pnls[t] = pnl_res["net_pnl"]
                    gross_pnls[t] = pnl_res["gross_pnl"]
                    total_costs[t] = pnl_res["total_tc"]
                    net_returns[t] = pnl_res["net_return"]
                    
                    self.risk.update_pnl_and_check_halts(pnl_res["net_pnl"])
                    
                    trade_records.append({
                        "event_idx": t,
                        "timestamp": df_lob["timestamp"].iloc[t] if "timestamp" in df_lob.columns else t,
                        "signal": dt,
                        "size": exec_size,
                        "entry_mid": entry_mid,
                        "exit_mid": exit_mid,
                        "gross_pnl": pnl_res["gross_pnl"],
                        "total_cost": pnl_res["total_tc"],
                        "net_pnl": pnl_res["net_pnl"],
                        "net_return": pnl_res["net_return"],
                        "is_filled": pnl_res.get("p_fill", 1.0) > 0.5
                    })

        trades_df = pd.DataFrame(trade_records) if len(trade_records) > 0 else pd.DataFrame(
            columns=["event_idx", "timestamp", "signal", "size", "entry_mid", "exit_mid", "gross_pnl", "total_cost", "net_pnl", "net_return", "is_filled"]
        )

        return {
            "trades_df": trades_df,
            "signals": signals[:num_events],
            "net_pnls": net_pnls[:num_events],
            "gross_pnls": gross_pnls[:num_events],
            "total_costs": total_costs[:num_events],
            "net_returns": net_returns[:num_events],
            "cumulative_pnl_series": np.cumsum(net_pnls[:num_events]),
            "num_trades": len(trades_df)
        }
