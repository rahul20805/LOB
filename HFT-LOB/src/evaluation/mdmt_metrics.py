"""
Multi-Dimensional Metric Taxonomy (MDMT) Evaluation Suite.
Implements Section 5 & Table 3:
- Tier A: Statistical Metrics (Accuracy, Weighted F1, Cohen's Kappa, RMSE)
- Tier B: Signal Quality Metrics (Directional Accuracy DA, Transaction Hit Rate THR, Information Coefficient IC, Turnover TO)
- Tier C: Trade-Level Financial Metrics (Hit Rate HR, Average Win/Loss, Profit Factor PF, Expectancy)
- Tier D: Portfolio-Level Financial Metrics (Cumulative P&L, Sharpe S, Sortino SD, Calmar C, Max Drawdown MDD)
- Tier E: Regime-Conditional Metrics (Regime Sharpe S^(R), Rolling IC)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error


class MDMTEvaluator:
    """
    Evaluates complete 5-tier metric stack for LOB predictive models.
    """
    @staticmethod
    def evaluate_tier_a_statistical(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Tier A: Statistical ML Metrics"""
        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        return {
            "accuracy": float(acc),
            "weighted_f1": float(f1_w),
            "macro_f1": float(f1_macro),
            "cohen_kappa": float(kappa)
        }

    @staticmethod
    def evaluate_tier_b_signal_quality(
        signals: np.ndarray,
        actual_returns: np.ndarray,
        net_returns: np.ndarray
    ) -> Dict[str, float]:
        """Tier B: Signal Quality Metrics"""
        non_zero = signals != 0
        if np.sum(non_zero) == 0:
            return {
                "directional_accuracy": 0.0,
                "transaction_hit_rate": 0.0,
                "information_coefficient": 0.0,
                "signal_turnover": 0.0
            }

        # Directional accuracy restricted to non-zero signals
        pred_dir = np.sign(signals[non_zero])
        act_dir = np.sign(actual_returns[non_zero])
        da = np.mean(pred_dir == act_dir)

        # Equation (16): Transaction Hit Rate (THR)
        thr = np.mean(net_returns[non_zero] > 0)

        # Information Coefficient (IC)
        valid_mask = np.isfinite(signals) & np.isfinite(actual_returns)
        if np.sum(valid_mask) > 5 and np.std(signals[valid_mask]) > 1e-6 and np.std(actual_returns[valid_mask]) > 1e-6:
            ic = float(np.corrcoef(signals[valid_mask], actual_returns[valid_mask])[0, 1])
        else:
            ic = 0.0

        # Signal Turnover: TO = 1/T * sum(|d_t - d_{t-1}|)
        turnover = float(np.mean(np.abs(np.diff(signals))))

        return {
            "directional_accuracy": float(da),
            "transaction_hit_rate": float(thr),
            "information_coefficient": float(ic),
            "signal_turnover": float(turnover)
        }

    @staticmethod
    def evaluate_tier_c_trade_level(net_returns: np.ndarray, signals: np.ndarray) -> Dict[str, float]:
        """Tier C: Trade-Level Financial Metrics"""
        active_returns = net_returns[signals != 0]
        if len(active_returns) == 0:
            return {
                "hit_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0
            }

        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        
        hr = len(wins) / len(active_returns)
        avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_l = float(np.mean(losses)) if len(losses) > 0 else 0.0
        
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        pf = gross_profit / gross_loss
        
        expectancy = hr * avg_w + (1.0 - hr) * avg_l

        return {
            "hit_rate": float(hr),
            "avg_win": float(avg_w),
            "avg_loss": float(avg_l),
            "profit_factor": float(pf),
            "expectancy": float(expectancy)
        }

    @staticmethod
    def evaluate_tier_d_portfolio_level(
        net_pnls: np.ndarray,
        annualization_factor: float = 252.0 * 390.0 # Intraday annualization factor
    ) -> Dict[str, float]:
        """
        Tier D: Portfolio-Level Financial Metrics (Eq 17 - 21).
        """
        cum_pnl = np.cumsum(net_pnls)
        total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0
        
        if len(net_pnls) == 0 or np.std(net_pnls) < 1e-8:
            return {
                "cumulative_pnl": total_pnl,
                "annualized_sharpe": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0
            }

        mean_r = np.mean(net_pnls)
        std_r = np.std(net_pnls)
        
        # Annualized Sharpe (Eq 18)
        sharpe = (mean_r / std_r) * np.sqrt(annualization_factor)
        
        # Downside deviation for Sortino (Eq 19)
        downside = net_pnls[net_pnls < 0]
        downside_std = np.std(downside) if len(downside) > 1 else std_r
        sortino = (mean_r / (downside_std + 1e-6)) * np.sqrt(annualization_factor)
        
        # Maximum Drawdown (Eq 21)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Calmar Ratio (Eq 20)
        calmar = (mean_r * annualization_factor) / (max_dd + 1e-6) if max_dd > 0 else 0.0

        return {
            "cumulative_pnl": float(total_pnl),
            "annualized_sharpe": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd)
        }

    @staticmethod
    def evaluate_tier_e_regime_conditional(
        net_pnls: np.ndarray,
        regimes: np.ndarray
    ) -> Dict[str, float]:
        """Tier E: Regime-Conditional Metrics (Eq 22)"""
        regime_sharpes = {}
        unique_regimes = np.unique(regimes)
        
        for reg in ["HV-T", "HV-M", "LV-T", "LV-M"]:
            mask = regimes == reg
            if np.sum(mask) > 5:
                reg_pnl = net_pnls[mask]
                std_r = np.std(reg_pnl)
                sh = (np.mean(reg_pnl) / (std_r + 1e-6)) * np.sqrt(252 * 390)
                regime_sharpes[f"sharpe_{reg}"] = float(sh)
            else:
                regime_sharpes[f"sharpe_{reg}"] = 0.0
                
        return regime_sharpes
