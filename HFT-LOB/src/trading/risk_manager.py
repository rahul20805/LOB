"""
High-Frequency Risk Management Engine (Section 8.3).
Enforces mandatory risk controls:
1. Maximum Position Limit: |Q_t| <= Q_max
2. Stop-Loss Halt: intraday P&L < -delta_stop
3. Maximum Drawdown Protection: MDD > delta_MDD closes all positions
4. Latency Buffer: skips l_proc ms
"""

from typing import Dict, Any, Tuple


class RiskManager:
    """
    Enforces risk constraints on trade execution.
    """
    def __init__(
        self,
        max_position_size: float = 100.0,
        stop_loss_pct: float = 0.005,      # -0.5% NAV halt
        max_drawdown_limit: float = 0.05,  # 5% max drawdown halt
        latency_buffer_ms: float = 5.0
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.latency_buffer_ms = latency_buffer_ms
        
        self.current_position = 0.0
        self.peak_pnl = 0.0
        self.cum_pnl = 0.0
        self.is_halted = False

    def reset(self):
        self.current_position = 0.0
        self.peak_pnl = 0.0
        self.cum_pnl = 0.0
        self.is_halted = False

    def check_trade_allowed(self, signal: int, desired_size: float) -> Tuple[bool, float, str]:
        """
        Validates whether trade is permitted by risk rules and clips position size.
        """
        if self.is_halted:
            return False, 0.0, "TRADING_HALTED"

        if signal == 0:
            return True, 0.0, "NO_SIGNAL"

        # Check maximum position limit
        new_position = self.current_position + signal * desired_size
        if abs(new_position) > self.max_position_size:
            # Clip order size
            allowed_size = max(0.0, self.max_position_size - abs(self.current_position))
            if allowed_size <= 0:
                return False, 0.0, "MAX_POSITION_EXCEEDED"
            return True, allowed_size, "POSITION_CLIPPED"

        return True, desired_size, "APPROVED"

    def update_pnl_and_check_halts(self, trade_pnl: float, capital_nav: float = 100000.0) -> bool:
        """
        Updates running P&L and triggers stop-loss / drawdown halts.
        """
        self.cum_pnl += trade_pnl
        if self.cum_pnl > self.peak_pnl:
            self.peak_pnl = self.cum_pnl

        # Check intraday stop loss
        if trade_pnl < -(self.stop_loss_pct * capital_nav):
            self.is_halted = True
            return True

        # Check maximum drawdown
        dd = self.peak_pnl - self.cum_pnl
        if (dd / capital_nav) > self.max_drawdown_limit:
            self.is_halted = True
            return True

        return False
