# Mathematical Formulation: LOB-PROFIT Framework

This document details all mathematical formulations from Chaudhary & Kushwaha (2024), providing their mathematical notation, physical meaning, Python implementation, numerical stability considerations, and pipeline placement.

---

## 1. Limit Order Book (LOB) Representation (Eq. 1 & 2)

### Mathematical Formulation
A Limit Order Book with $K$ price levels is represented at time $t$ as:
$$\mathcal{L}_t = \left\{ (P_{k,t}^a, V_{k,t}^a), (P_{k,t}^b, V_{k,t}^b) \right\}_{k=1}^K$$
where $P_{k,t}^a, V_{k,t}^a$ are the ask price and volume at depth level $k$, and $P_{k,t}^b, V_{k,t}^b$ are the bid price and volume at level $k$.

The bid-ask spread $s_t$ and mid-price $m_t$ are defined as:
$$s_t = P_{1,t}^a - P_{1,t}^b > 0$$
$$m_t = \frac{P_{1,t}^a + P_{1,t}^b}{2}$$

### Python Implementation
```python
def compute_spread_and_mid(best_ask: float, best_bid: float) -> tuple[float, float]:
    spread = best_ask - best_bid
    mid = 0.5 * (best_ask + best_bid)
    return spread, mid
```

---

## 2. Smoothed Return & Ternary Directional Target (Eq. 3 & 4)

### Mathematical Formulation
To eliminate microstructural noise, the mid-price is smoothed over a backward rolling window $H$:
$$\bar{m}_t = \frac{1}{H} \sum_{h=0}^{H-1} m_{t-h}$$

The forward smoothed relative return over prediction horizon $\Delta$ is:
$$l_t = \frac{\bar{m}_{t+\Delta} - \bar{m}_t}{\bar{m}_t}$$

Given threshold $\alpha > 0$, the ternary target $y_t \in \{-1, 0, +1\}$ (down, stationary, up) is:
$$y_t = \begin{cases} +1 & \text{if } l_t > \alpha \\ -1 & \text{if } l_t < -\alpha \\ 0 & \text{otherwise} \end{cases}$$

### Numerical Stability
- Ensure $\bar{m}_t > 0$.
- Calibrate $\alpha$ on the training set to ensure non-degenerate class balance across $\{-1, 0, +1\}$.

---

## 3. Order Flow Imbalance (OFI) (Eq. 5, 6, 7)

### Mathematical Formulation
Following Cont et al. (2023), the level-$k$ bid volume update $\Delta W_{k,t}^b$ and ask volume update $\Delta W_{k,t}^a$ are:
$$\Delta W_{k,t}^b = V_{k,t}^b \mathbf{1}_{\{P_{k,t}^b \ge P_{k,t-1}^b\}} - V_{k,t-1}^b \mathbf{1}_{\{P_{k,t}^b \le P_{k,t-1}^b\}}$$
$$\Delta W_{k,t}^a = V_{k,t}^a \mathbf{1}_{\{P_{k,t}^a \le P_{k,t-1}^a\}} - V_{k,t-1}^a \mathbf{1}_{\{P_{k,t}^a \ge P_{k,t-1}^a\}}$$

The level-$k$ Order Flow Imbalance is:
$$\text{OFI}_t^{(k)} = \Delta W_{k,t}^b - \Delta W_{k,t}^a$$

---

## 4. Multi-Level OFI PCA & Cumulative Imbalance Index (Eq. 12, 13, 14)

### Mathematical Formulation
The multi-level OFI vector is:
$$\mathbf{OFI}_t = \left( \text{OFI}_t^{(1)}, \dots, \text{OFI}_t^{(K)} \right)^\top \in \mathbb{R}^K$$

The integrated OFI via first principal component $\mathbf{u}_1$ is:
$$\text{OFI}_t^{int} = \mathbf{u}_1^\top \mathbf{OFI}_t$$

The Cumulative Imbalance Index ($CII$) over rolling window $W$ is:
$$CII_t = \sum_{\tau = t-W}^t \text{OFI}_\tau^{int}$$

---

## 5. Basic Microstructure & Dynamic Features (Eq. 8, 9, 10, 11)

### Mathematical Formulation
- **Volume Imbalance**:
  $$\psi_t = \frac{\sum_k V_{k,t}^b - \sum_k V_{k,t}^a}{\sum_k V_{k,t}^b + \sum_k V_{k,t}^a}$$
- **Weighted Mid-Price (Microprice)**:
  $$m_t^w = \frac{P_{1,t}^b V_{1,t}^a + P_{1,t}^a V_{1,t}^b}{V_{1,t}^a + V_{1,t}^b}$$
- **Price Momentum**: $\rho_t^{(\ell)} = m_t - m_{t-\ell}$ for $\ell \in \{1, 5, 10, 30\}$.
- **Spread Dynamics**: $\Delta s_t = s_t - s_{t-1}$.
- **Realized Volatility**: $\hat{\sigma}_t = \sqrt{\sum_{\tau=t-W}^t (m_\tau - m_{\tau-1})^2}$.
- **Queue Dynamics**: $\Delta V_{1,t}^b = V_{1,t}^b - V_{1,t-1}^b$, $\Delta V_{1,t}^a = V_{1,t}^a - V_{1,t-1}^a$.

---

## 6. Signal Conversion & Transaction Hit Rate (Eq. 15 & 16)

### Mathematical Formulation
Given model probability output $\hat{\mathbf{p}}_t = (\hat{p}_t^{-1}, \hat{p}_t^0, \hat{p}_t^{+1})$, the trading signal $d_t \in \{-1, 0, +1\}$ is:
$$d_t = \begin{cases} 
+1 & \text{if } \hat{p}_t^{+1} > \theta(R_t) \text{ and } \hat{p}_t^{+1} = \max(\hat{\mathbf{p}}_t) \\
-1 & \text{if } \hat{p}_t^{-1} > \theta(R_t) \text{ and } \hat{p}_t^{-1} = \max(\hat{\mathbf{p}}_t) \\
0 & \text{otherwise}
\end{cases}$$

The **Transaction Hit Rate (THR)**:
$$\text{THR} = \frac{\#\{t : d_t \neq 0 \text{ and } r_t^{net} > 0\}}{\#\{t : d_t \neq 0\}}$$

---

## 7. Realistic Execution Cost Integration Model (RECIM) (Eq. 23 - 28)

### Mathematical Formulation
Net return per trade of size $Q$:
$$r_t^{net} = d_t \cdot (m_{t+\Delta} - m_t) - TC(|Q|, t)$$

Total Transaction Cost Decomposition:
$$TC(|Q|, t) = \frac{s_t}{2} + f \cdot P_t + I(|Q|, t)$$

Crossover Market Impact:
$$I(|Q|, t) = \sigma_t \cdot f\left( \frac{|Q|}{V_t^{daily}} \right)$$
$$f(\xi) \approx \begin{cases} \eta \xi & \text{if } \xi \le \xi^* \\ \eta \sqrt{\xi^*} \sqrt{\xi} & \text{if } \xi > \xi^* \end{cases}$$
where $\xi^* \approx 10^{-3}$ and $\eta$ is calibrated to the asset liquidity.

Limit Order Fill Probability (Complementary Survival Function):
$$p_{fill}^{(k)}(\Delta) = 1 - \exp\left( - \int_0^\Delta \lambda^{(k)}(u) du \right)$$

---

## 8. MDMT Portfolio-Level Financial Metrics (Eq. 17 - 21)

### Mathematical Formulation
- **Cumulative P&L**: $\Pi_T = \sum_{t=1}^T r_t^{net} Q_t$
- **Annualized Sharpe Ratio**: $S = \frac{\bar{R} - r_f}{\sigma_R} \sqrt{252}$ (or $\sqrt{N_{periods}}$ for intraday ticks)
- **Sortino Ratio**: $S_D = \frac{\bar{R} - r_f}{\sigma_D} \sqrt{252}$ where $\sigma_D = \sqrt{\mathbb{E}[\min(R_t - r_f, 0)^2]}$
- **Maximum Drawdown (MDD)**:
  $$\text{MDD} = \max_{t \in [0,T]} \frac{\Pi_{0:t}^{max} - \Pi_t}{\Pi_{0:t}^{max}}$$
- **Calmar Ratio**: $C = \frac{\bar{R} \cdot 252}{\text{MDD}}$
- **Regime-Conditional Sharpe**: $S^{(R)} = \text{Sharpe}\left( \{r_t^{net} : \text{regime}(t) = R\} \right)$

---

## 9. Regime-Adaptive Evaluation Protocol (RAEP) (Eq. 29 & 30)

### Mathematical Formulation
4-Quadrant Regime Identification:
$$R_t = \begin{cases}
\text{HV-T} & \sigma_t > \bar{\sigma} \text{ and } \rho_t > \bar{\rho} \quad \text{(High-vol trending)} \\
\text{HV-M} & \sigma_t > \bar{\sigma} \text{ and } \rho_t \le \bar{\rho} \quad \text{(High-vol mean-reverting)} \\
\text{LV-T} & \sigma_t \le \bar{\sigma} \text{ and } \rho_t > \bar{\rho} \quad \text{(Low-vol trending)} \\
\text{LV-M} & \sigma_t \le \bar{\sigma} \text{ and } \rho_t \le \bar{\rho} \quad \text{(Low-vol mean-reverting)}
\end{cases}$$

Performance Degradation Index (PDI):
$$\text{PDI}(M) = \frac{\max(0, M(R_{train}) - M(R_{test}))}{|M(R_{train})| + \epsilon}$$
