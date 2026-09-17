"""
Publication-Quality Visualization Suite for LOB-PROFIT Research.
Generates Figures 1 through 15 at 300 DPI conforming to academic standards.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


class PublicationPlotter:
    """
    Generates all 15 publication figures for the LOB-PROFIT study.
    """
    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_figure_5_dataset_distribution(self, df_lob: pd.DataFrame, save_name: str = "figure_05_dataset_distribution.png"):
        """Figure 5: Dataset distribution, spread dynamics, and volume imbalance."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. Spread distribution
        if "spread" in df_lob.columns:
            sns.histplot(df_lob["spread"], kde=True, ax=axes[0, 0], color="#1f77b4")
            axes[0, 0].set_title("A. Bid-Ask Spread Distribution")
            axes[0, 0].set_xlabel("Spread")
            axes[0, 0].set_ylabel("Count")

        # 2. Volume Imbalance
        if "volume_imbalance_topK" in df_lob.columns:
            sns.histplot(df_lob["volume_imbalance_topK"], kde=True, ax=axes[0, 1], color="#2ca02c")
            axes[0, 1].set_title("B. Top-K Volume Imbalance (psi_t)")
            axes[0, 1].set_xlabel("Volume Imbalance")
            axes[0, 1].set_ylabel("Density")
        elif "bid_v_1" in df_lob.columns:
            imb = (df_lob["bid_v_1"] - df_lob["ask_v_1"]) / (df_lob["bid_v_1"] + df_lob["ask_v_1"] + 1e-6)
            sns.histplot(imb, kde=True, ax=axes[0, 1], color="#2ca02c")
            axes[0, 1].set_title("B. L1 Volume Imbalance")
            axes[0, 1].set_xlabel("Volume Imbalance")

        # 3. Mid Price Evolution
        if "mid_price" in df_lob.columns:
            axes[1, 0].plot(df_lob["mid_price"].iloc[:500].values, color="#d62728", lw=1.2)
            axes[1, 0].set_title("C. Mid-Price Microstructure Path (First 500 Events)")
            axes[1, 0].set_xlabel("Event Index")
            axes[1, 0].set_ylabel("Mid Price")

        # 4. Target Class Balance
        if "target_label" in df_lob.columns:
            counts = df_lob["target_label"].value_counts().sort_index()
            labels = ["Down (-1)", "Stationary (0)", "Up (+1)"]
            axes[1, 1].bar(labels, [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)], color=["#e74c3c", "#95a5a6", "#2ecc71"])
            axes[1, 1].set_title("D. Ternary Target Distribution (H=5, Delta=10)")
            axes[1, 1].set_ylabel("Sample Count")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_6_model_comparison(self, metrics_df: pd.DataFrame, save_name: str = "figure_06_model_performance_comparison.png"):
        """Figure 6: Model Accuracy vs Realized Net Sharpe vs P&L (Exposing accuracy-profitability disconnect)."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        models = metrics_df["model"].values
        acc = metrics_df["accuracy"].values
        sharpe = metrics_df["annualized_sharpe"].values
        pnl = metrics_df["cumulative_pnl"].values

        # Accuracy
        axes[0].barh(models, acc, color="#3498db")
        axes[0].set_title("A. Statistical Accuracy (Tier A)")
        axes[0].set_xlabel("Accuracy")
        axes[0].set_xlim(0.0, 1.0)

        # Sharpe
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpe]
        axes[1].barh(models, sharpe, color=colors)
        axes[1].axvline(0, color="black", linestyle="--", lw=0.8)
        axes[1].set_title("B. Net Annualized Sharpe (Tier D)")
        axes[1].set_xlabel("Sharpe Ratio (Net of RECIM)")

        # Cumulative P&L
        pnl_colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnl]
        axes[2].barh(models, pnl, color=pnl_colors)
        axes[2].axvline(0, color="black", linestyle="--", lw=0.8)
        axes[2].set_title("C. Net Cumulative P&L (Tier D)")
        axes[2].set_xlabel("Net Realized P&L ($)")

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_7_confusion_matrices(self, conf_matrices: Dict[str, np.ndarray], save_name: str = "figure_07_confusion_matrices.png"):
        """Figure 7: Confusion matrices for key models."""
        n_models = len(conf_matrices)
        fig, axes = plt.subplots(1, max(n_models, 2), figsize=(4 * max(n_models, 2), 3.5))
        
        classes = ["Down", "Stat", "Up"]
        for idx, (m_name, cm) in enumerate(conf_matrices.items()):
            ax = axes[idx] if n_models > 1 else axes
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title(f"{m_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_11_equity_curves(self, equity_dict: Dict[str, np.ndarray], save_name: str = "figure_11_trading_performance.png"):
        """Figure 11: Realized Cumulative Net P&L Equity Curves across models."""
        plt.figure(figsize=(10, 5))
        for m_name, pnl_curve in equity_dict.items():
            plt.plot(pnl_curve, label=m_name, lw=1.5)
            
        plt.axhline(0, color="black", linestyle="--", lw=0.8)
        plt.title("Figure 11: Realized Cumulative Equity Curves (Net of RECIM Costs)")
        plt.xlabel("Trade Execution Event Index")
        plt.ylabel("Net Cumulative P&L ($)")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_12_drawdown_curves(self, equity_dict: Dict[str, np.ndarray], save_name: str = "figure_12_drawdown_dynamics.png"):
        """Figure 12: Underwater Drawdown Dynamics."""
        plt.figure(figsize=(10, 5))
        for m_name, pnl_curve in equity_dict.items():
            running_max = np.maximum.accumulate(pnl_curve)
            dd = running_max - pnl_curve
            plt.plot(-dd, label=m_name, lw=1.5)
            
        plt.title("Figure 12: Underwater Drawdown Curves")
        plt.xlabel("Trade Execution Event Index")
        plt.ylabel("Drawdown ($)")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_13_latency(self, latency_df: pd.DataFrame, save_name: str = "figure_13_latency_distribution.png"):
        """Figure 13: Inference Latency vs Throughput."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        models = latency_df["model"].values
        p95_lat = latency_df["p95_latency_ms"].values
        throughput = latency_df["throughput_samples_per_sec"].values

        axes[0].bar(models, p95_lat, color="#9b59b6")
        axes[0].set_title("A. 95th Percentile Inference Latency (ms)")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].axhline(5.0, color="red", linestyle="--", label="5ms Latency Budget")
        axes[0].legend()

        axes[1].bar(models, throughput, color="#1abc9c")
        axes[1].set_title("B. Inference Throughput (Samples / sec)")
        axes[1].set_ylabel("Throughput")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_10_ablation_study(self, ablation_df: pd.DataFrame, save_name: str = "figure_10_ablation_study.png"):
        """Figure 10: Ablation study showing Sharpe change when features/components are removed."""
        plt.figure(figsize=(9, 4.5))
        components = ablation_df["component_removed"].values
        sharpe_delta = ablation_df["sharpe_change"].values
        
        colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in sharpe_delta]
        plt.barh(components, sharpe_delta, color=colors)
        plt.axvline(0, color="black", linestyle="--", lw=0.8)
        plt.title("Figure 10: Ablation Study — Impact on Net Annualized Sharpe")
        plt.xlabel("Change in Net Sharpe Ratio (Delta S)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_figure_14_regimes(self, regime_sharpes_df: pd.DataFrame, save_name: str = "figure_14_regime_generalization.png"):
        """Figure 14: Cross-regime Sharpe comparison and PDI analysis."""
        plt.figure(figsize=(10, 5))
        regimes = ["HV-T", "HV-M", "LV-T", "LV-M"]
        models = regime_sharpes_df["model"].unique()
        
        x = np.arange(len(regimes))
        width = 0.8 / len(models)
        
        for i, m in enumerate(models):
            sub = regime_sharpes_df[regime_sharpes_df["model"] == m]
            vals = [sub[f"sharpe_{r}"].values[0] if f"sharpe_{r}" in sub else 0.0 for r in regimes]
            plt.bar(x + i * width, vals, width, label=m)
            
        plt.axhline(0, color="black", linestyle="--", lw=0.8)
        plt.xticks(x + width * (len(models) - 1) / 2, regimes)
        plt.title("Figure 14: Regime-Conditional Sharpe Ratios Across Market States")
        plt.xlabel("Market Regime (RAEP 4-Quadrant Taxonomy)")
        plt.ylabel("Annualized Sharpe Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, save_name)
        plt.savefig(out_path)
        plt.close()
        return out_path
