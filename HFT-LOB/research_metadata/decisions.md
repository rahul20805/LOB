# Research Engineering Decisions Log

## 1. Zero-Leakage Preprocessing
- **Decision**: All normalization statistics ($\mu, \sigma$) must be computed strictly via expanding windows ($\tilde{x}_t = (x_t - \mu_{1:t-1})/\sigma_{1:t-1}$) or fitted exclusively on the training split $m-W \dots m-1$ and applied to test fold $m$.
- **Rationale**: Standard batch Z-score normalizers that use global or test-set statistics leak future distribution parameters into the past, artificially inflating classification accuracy and out-of-sample Sharpe ratios.

## 2. Multi-Level OFI and PCA Dimensionality Reduction
- **Decision**: We compute piecewise level-$k$ Order Flow Imbalance (OFI) across all tracked levels $k=1\dots K$, then estimate the first principal component projection vector $\mathbf{u}_1$ strictly on the training fold.
- **Rationale**: Following Cont et al. (2023), the first principal component accounts for >80% of total multi-level OFI variance, providing high signal-to-noise ratio while preventing multicollinearity.

## 3. RECIM Realistic Execution Cost Parameters
- **Decision**: In RECIM simulation, we incorporate:
  1. Full bid-ask spread crossing for aggressive market orders ($s_t/2$).
  2. Exchange fee rate $f = 2\text{ bps}$ ($0.0002$) for spot equities/crypto.
  3. Crossover market impact parameter $\eta = 0.5$, $\xi^* = 10^{-3}$, switching continuously between linear and square-root regimes.
  4. Fill hazard rate $\lambda(u)$ yielding realistic survival probabilities for passive limit orders.
  5. Latency buffer $\ell_{proc} = 5\text{ ms}$ (GPU/CPU inference delay).
- **Rationale**: Eliminates the "paper profit" illusion where high-frequency signals exploit micro-spreads that are completely non-executable after exchange fees and price impact.

## 4. Regime Partitioning (RAEP)
- **Decision**: Market regimes are classified dynamically into 4 quadrants (HV-T, HV-M, LV-T, LV-M) using rolling medians of 30-period realized volatility $\bar{\sigma}$ and first-order return autocorrelation $\bar{\rho}$.
- **Rationale**: Distinguishes trending markets (positive autocorrelation) from mean-reverting markets (negative autocorrelation) across high and low volatility regimes, enabling calculation of the Performance Degradation Index ($PDI$).

## 5. Multi-Dataset Integration Strategy
- **Decision**: Standardize feature representations into common dimensions across FI-2010 and Crypto datasets, evaluate both dataset-specific models and pooled multi-dataset models, and perform cross-market transfer testing.
- **Rationale**: Rigorously tests whether LOB representations learn universal microstructure dynamics or overfit to specific asset regimes.
