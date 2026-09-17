# Literature Review and Related Work Analysis

## 1. Executive Summary
This document analyzes twelve landmark research papers spanning 2019 to 2025 across high-frequency trading (HFT), deep learning on Limit Order Books (LOB), Order Flow Imbalance (OFI), market impact modeling, and algorithmic execution. It provides a formal literature gap analysis that clearly separates:
1. **What Existing Studies Have Accomplished**
2. **The Methodological Deficiencies in Existing Literature**
3. **What Our Study (LOB-PROFIT Framework) Provides**

---

## 2. Structured Analysis of Research Gaps (Gaps 1–6)

### Gap 1: Market Impact and Transaction Cost Modelling
- **Prior Work**: Bucci et al. (2019) proved the universal crossover from a linear regime for small orders ($\xi \le \xi^*$) to a square-root regime for large orders ($\xi > \xi^*$) using 8 million institutional equity trades. Arroyo et al. (2024) developed survival analysis for limit order fill lifetimes.
- **Deficiency**: Standard LOB deep learning literature (Zhang et al. 2019, Nousi et al. 2019) universally assumes zero transaction costs, instantaneous execution, and zero market impact.
- **LOB-PROFIT Contribution**: Systematizes RECIM (Realistic Execution Cost Integration Model), unifying half-spread crossing, exchange fees, empirical Bucci crossover impact, and limit order survival hazard rates into an event-driven backtesting execution handler.

### Gap 2: Order Flow Imbalance (OFI) as a Superior Input Representation
- **Prior Work**: Cont et al. (2023) defined piecewise multi-level OFI and demonstrated via PCA that the first principal component explains >80% of cross-sectional variance. Kolm et al. (2023) demonstrated that LSTMs trained on OFI outperform raw LOB models across 115 NASDAQ stocks.
- **Deficiency**: Deep learning models continue to rely heavily on raw unnormalized 2D price-volume grids without integrating stationary OFI dynamics.
- **LOB-PROFIT Contribution**: Integrates multi-level OFI and PCA projection ($OFI^{int}$) as a mandatory Tier 2 tier in a formal 4-tier feature hierarchy.

### Gap 3: Optimal Execution & Reinforcement Learning
- **Prior Work**: Guo et al. (2023) designed Attn-LOB RL agents; Gašperov & Kostanjčar (2022) trained deep RL controllers on multivariate Hawkes process simulators.
- **Deficiency**: RL approaches are highly sample-inefficient and restricted to simulated or stylized environments without evaluating classical ML or Transformer architectures under a standardized financial metric hierarchy.
- **LOB-PROFIT Contribution**: Standardizes the execution interface (Phase 5) allowing policy-based and supervised signal models to be evaluated under identical financial constraints.

### Gap 4: Profitability Evaluation & The Accuracy-Profitability Gap
- **Prior Work**: Briola et al. (2024) proved empirically on NASDAQ stocks that DeepLOB models with >60% accuracy generate zero cumulative P&L. Jabbar & Jalil (2024) evaluated 41 ML models on Bitcoin LOB data and proved that the Spearman rank correlation between statistical accuracy and Sharpe ratio is $\rho < 0.25$. Prata et al. (2024) benchmarked 15 deep learning models and found that all models suffer negative P&L once transaction costs are deducted.
- **Deficiency**: No prior framework existed that formalized a complete multi-tier metric taxonomy mapping statistical accuracy through signal quality to portfolio P&L and risk-adjusted ratios.
- **LOB-PROFIT Contribution**: Introduces the Multi-Dimensional Metric Taxonomy (MDMT), a 5-tier evaluation stack (Tiers A through E) and provides the 11-point practitioner checklist.

### Gap 5: Regime Robustness and Out-of-Sample Generalization
- **Prior Work**: Wong & Barahona (2023) developed GBDT with dropout and online Sharpe-weighted dynamic ensembling to preserve performance across tabular regime shifts.
- **Deficiency**: Deep models trained on static LOB datasets collapse when transferred out-of-sample due to liquidity and volatility clustering.
- **LOB-PROFIT Contribution**: Formalizes the Regime-Adaptive Evaluation Protocol (RAEP) featuring 4-quadrant market regime classification ($R_t \in \{\text{HV-T, HV-M, LV-T, LV-M}\}$), walk-forward validation across $M \ge 5$ folds, Performance Degradation Index ($PDI$), and real-time dynamic ensembling.

### Gap 6: Cryptocurrency LOB Microstructure
- **Prior Work**: Jha et al. (2020) applied temporal CNNs to Coinbase Bitcoin spot data (71% walk-forward accuracy). Camaglia et al. (2023) applied Hawkes processes to USDT/USD pairs.
- **Deficiency**: Studies lack standardized evaluation across both traditional equities and 24/7 crypto markets under realistic maker/taker fee structures.
- **LOB-PROFIT Contribution**: Evaluates cross-market transfer across traditional benchmark equities (FI-2010) and high-frequency cryptocurrency spot books (Binance BTC/USDT & ETH/USDT).
