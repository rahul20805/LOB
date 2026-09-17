# Paper Inventory: Beyond Accuracy: Profitability-Driven Evaluation of LOB-Based ML Trading Models

## 1. Bibliographic Metadata
- **Paper Title**: Beyond Accuracy: Profitability-Driven Evaluation of LOB-Based ML Trading Models
- **Authors**: Rahul Yadav
- **Corresponding Authors**: Rahul Yadav : rahul20806@gmail.com`)
- **Year**: 2026
- **Venue**: Working Paper / arXiv / Journal of Financial Data Science target
- **DOI**: N/A (Preprint / Working Paper under review)
- **Primary Keywords**: Limit Order Book, Machine Learning, Profitability Evaluation, Market Microstructure, Algorithmic Trading, High-Frequency Trading, Market Impact

## 2. Research Problem & Core Hypothesis
- **Problem Statement**: High classification accuracy on benchmark Limit Order Book (LOB) datasets (e.g., FI-2010, LOBSTER) does not translate into profitable trading strategies in live or realistic backtested environments. The mapping $A \nearrow \Rightarrow \Pi \nearrow$ (Accuracy increases implies Cumulative P&L increases) is empirically false.
- **Root Causes Identified**:
  1. Low signal-to-noise ratio in high-frequency financial time series.
  2. Transaction costs (bid-ask spread and exchange fees) and non-linear market impact that erode marginal statistical gains.
  3. Execution latency and non-trivial fill probabilities for passive limit orders.
  4. Market regime non-stationarity causing out-of-sample model degradation.
- **Core Hypothesis**: An end-to-end evaluation pipeline that explicitly integrates realistic execution costs (RECIM), multi-dimensional financial metrics (MDMT), and regime-adaptive walk-forward validation (RAEP) resolves the accuracy-profitability disconnect and provides a true economic assessment of LOB models.

## 3. The 7-Phase LOB-PROFIT Framework
1. **Phase 1: Data Acquisition & Preprocessing**: Clean auctions/outliers ($\pm 5\sigma$), synchronize to time/volume bars, expanding-window Z-score normalization $\tilde{x}_t = (x_t - \mu_{1:t-1})/\sigma_{1:t-1}$, intraday seasonality removal $\sigma^2(\tau)$, and smoothed mid-price return label construction $l_t = (\bar{m}_{t+\Delta} - \bar{m}_t)/\bar{m}_t$ with threshold $\alpha$.
2. **Phase 2: Feature Engineering Taxonomy**:
   - **Tier 0**: Raw LOB state ($P_k^b, V_k^b, P_k^a, V_k^a$ for $k=1\dots K$).
   - **Tier 1**: Basic Microstructure features (spread $s_t$, mid-price $m_t$, volume imbalance $\psi_t$, weighted mid / microprice $m_t^w$).
   - **Tier 2**: Order Flow Imbalance (multi-level OFI vector $OFI_t$, PCA first component projection $OFI^{int}_t$, Cumulative Imbalance Index $CII_t$).
   - **Tier 3**: Time-Sensitive Dynamic Features (multi-horizon price momentum $\rho_t^{(\ell)}$, spread dynamics $\Delta s_t$, realized volatility $\hat{\sigma}_t$, time-of-day sinusoidal encoding, queue dynamics $\Delta V_{1,t}^b, \Delta V_{1,t}^a$).
3. **Phase 3: Model Selection & Training**:
   - Classical Baselines: Logistic Regression, Linear Regression, Kernel SVM, Random Forest, XGBoost / GBDT with dropout.
   - Deep Learning Architectures: DeepLOB Spatial CNN, DeepLOB Spatiotemporal CNN+LSTM, LiT / TransLOB Transformer.
   - Training Guidelines: Strict chronological walk-forward, class-weighted cross-entropy, dropout ($p \in [0.2, 0.5]$), L2 weight decay, OFI pre-training.
4. **Phase 4: Signal-to-Trade Conversion**: Dynamic confidence thresholding $\theta(R_t)$ conditioned on market regime $R_t$; calculation of Transaction Hit Rate (THR).
5. **Phase 5: Realistic Execution Cost Integration Model (RECIM)**: Total transaction cost $TC(|Q|, t) = s_t/2 + f \cdot P_t + I(|Q|, t)$ with crossover linear-to-square-root market impact $I(|Q|, t) = \sigma_t f(|Q|/V_t^{daily})$ and limit order survival fill probability $p_{fill}(\Delta) = 1 - \exp(-\int_0^\Delta \lambda(u)du)$.
6. **Phase 6: Multi-Dimensional Metric Taxonomy (MDMT)**:
   - **Tier A**: Statistical Metrics (Accuracy, Weighted F1, Cohen's Kappa, RMSE).
   - **Tier B**: Signal Quality Metrics (Directional Accuracy DA, Transaction Hit Rate THR, Information Coefficient IC, Signal Turnover TO).
   - **Tier C**: Trade-Level Financial Metrics (Hit Rate HR, Average Win/Loss $\bar{W}/\bar{L}$, Profit Factor PF, Expectancy $E[r^{net}]$).
   - **Tier D**: Portfolio-Level Financial Metrics (Cumulative P&L $\Pi_T$, Annualized Sharpe $S$, Sortino $S_D$, Calmar $C$, Max Drawdown $MDD$).
   - **Tier E**: Regime-Conditional Metrics (Regime-conditional Sharpe $S^{(R)}$, Rolling IC).
7. **Phase 7: Regime-Adaptive Evaluation Protocol (RAEP)**:
   - 4-quadrant regime partitioning: High-Vol Trending (HV-T), High-Vol Mean-Reverting (HV-M), Low-Vol Trending (LV-T), Low-Vol Mean-Reverting (LV-M) via rolling medians $(\bar{\sigma}, \bar{\rho})$.
   - Rolling walk-forward validation across $M$ folds.
   - Performance Degradation Index $PDI(M) = \max(0, M(R_{train}) - M(R_{test})) / (|M(R_{train})| + \epsilon)$.
   - Online adaptation: dynamic model ensembling and model retraining when $PDI > \epsilon_{PDI}$.

## 4. Key References & Benchmarks Cited
- [1] Briola et al. (2024) - Deep limit order book forecasting: A microstructural guide.
- [2] Zhang, Zohren, Roberts (2019) - DeepLOB: Deep convolutional neural networks for limit order books (IEEE TSP).
- [3] Nousi et al. (2019) - Machine learning for forecasting mid price movement using limit order book data.
- [5] Bucci et al. (2019) - Crossover from linear to square-root market impact (PRL).
- [6] Arroyo et al. (2024) - Execution and cancellation lifetimes in foreign currency markets.
- [7] Cont, Cucuringu, Zhang (2023) - Cross-impact of order flow imbalance in equity markets (Quantitative Finance).
- [8] Kolm, Turiel, Westray (2023) - Deep order flow imbalance: Extracting alpha at multiple horizons (Mathematical Finance).
- [9] Guo, Tao, Wen (2023) - Market making with deep reinforcement learning from limit order books.
- [10] Gašperov, Kostanjčar (2022) - Deep reinforcement learning for market making under Hawkes process model.
- [11] Jabbar, Jalil (2024) - Profitability-driven evaluation of machine learning models in cryptocurrency trading.
- [13] Prata et al. (2024) - LOB-based deep learning models for stock price trend prediction: A benchmark study.
- [14] Wong, Barahona (2023) - Online learning techniques for prediction of temporal tabular datasets with regime changes.
- [15] Jha et al. (2020) - Deep learning for digital asset limit order books.
- [18] Li, Zhang, Xiao (2025) - LiT: Limit order book transformer.
- [20] Maglaras, Moallemi, Zheng (2021) - Optimal execution in a limit order book and associated market impact model.
