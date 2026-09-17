# LOB-PROFIT: Profitability-Driven Evaluation of LOB-Based ML Trading Models

**Authors**: Rahul Yadav 
**Paper**: *Beyond Accuracy: Profitability-Driven Evaluation of LOB-Based ML Trading Models* (2026)

---

## 📌 Executive Overview
High classification accuracy on Limit Order Book (LOB) benchmarks routinely fails to yield profitable trading strategies in live or realistic backtested environments. This codebase implements the complete, reproducible **LOB-PROFIT Framework**, spanning seven formal phases:
1. **Data Acquisition & Preprocessing**: Outlier removal ($\pm 5\sigma$), synchronization, expanding-window Z-score normalization (zero look-ahead), and calibrated smoothed ternary return labels ($y_t \in \{-1, 0, +1\}$).
2. **4-Tier Feature Engineering Taxonomy**:
   - Tier 0: Raw 2D LOB states ($P_k^b, V_k^b, P_k^a, V_k^a$)
   - Tier 1: Microstructure features (spread, mid-price, volume imbalance $\psi_t$, microprice $m_t^w$)
   - Tier 2: Order Flow Imbalance ($\text{OFI}_t^{(k)}$, PCA-integrated $\text{OFI}_t^{int}$, Cumulative Imbalance Index $CII_t$)
   - Tier 3: Time-sensitive dynamics ($\rho_t^{(\ell)}$, $\Delta s_t$, realized volatility $\hat{\sigma}_t$, cyclical time encodings)
3. **Model Selection & Architecture Zoo**:
   - Baselines: Logistic Regression, Random Forest, XGBoost / GBDT with dropout, Kernel SVM
   - Deep Learning: DeepLOB Spatial CNN, DeepLOB Spatiotemporal CNN+LSTM, LiT (Limit Order Book Transformer)
   - Ensembles: Dynamic Sharpe-Weighted Ensemble
4. **Signal-to-Trade Conversion**: Dynamic confidence thresholding $\theta(R_t)$ and profit filter.
5. **Realistic Execution Cost Integration Model (RECIM)**: Total transaction cost accounting ($s_t/2 + f \cdot P_t + I(|Q|, t)$), Bucci crossover square-root market impact, and limit order survival fill probabilities.
6. **Multi-Dimensional Metric Taxonomy (MDMT)**: 5-Tier evaluation hierarchy (Tiers A to E: Statistical, Signal Quality, Trade-Level, Portfolio-Level P&L/Sharpe/Sortino/Calmar, Regime-Conditional).
7. **Regime-Adaptive Evaluation Protocol (RAEP)**: 4-Quadrant market regime classification (HV-T, HV-M, LV-T, LV-M), rolling walk-forward CV, Performance Degradation Index ($PDI$), and online adaptation.

---

## 🚀 Quickstart & Reproduction

### Option A: One-Click Jupyter Notebook (Kaggle & Google Colab)
Open and run [HFT_LOB_PROFIT_RESEARCH_PIPELINE.ipynb](file:///d:/ANANT/hhfft/HFT_LOB_PROFIT_RESEARCH_PIPELINE.ipynb).  
It is completely self-contained, contains all data loaders, model definitions, RECIM engines, MDMT evaluators, and plots all publication figures with zero external file dependencies!

### Option B: Command-Line Pipeline Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit & integration test suite
python -m pytest tests/

# 3. Execute complete end-to-end research pipeline
python run_pipeline.py --config configs/full_experiment.yaml
```

---

## 📂 Project Structure
```
d:/ANANT/hhfft/
├── HFT_LOB_PROFIT_RESEARCH_PIPELINE.ipynb  # Self-contained standalone Kaggle/Colab notebook
├── run_pipeline.py                         # Complete research pipeline orchestrator CLI
├── requirements.txt                        # Python dependencies
├── environment.yml                         # Conda environment specification
├── CITATIONS.md                            # Complete academic citations
├── LICENSE_NOTES.md                        # Licensing, data provenance & ethics notes
├── HFT_RESEARCH_COMPLETE.zip               # Complete research archive
├── HFT_RESULTS_ONLY.zip                    # Compact results, charts & tables archive
│
├── datasets/                               # Real academic & crypto LOB data
│   ├── raw/
│   ├── processed/
│   └── checksums/
│       └── checksums.csv                   # SHA256 verified hashes
│
├── src/                                    # Modular source library
│   ├── math/                               # Formulas (Eq 1-14, 25-26, 29-30) & RECIM engine
│   ├── data/                               # Downloader & zero-leakage preprocessor
│   ├── features/                           # 4-Tier feature engineering taxonomy
│   ├── targets/                            # Ternary direction target builder
│   ├── models/                             # Baselines, DeepLOB, LiT Transformer & Ensembles
│   ├── training/                           # Walk-forward CV & RAEP regime detector
│   ├── evaluation/                         # MDMT 5-tier metrics & statistical tests
│   ├── trading/                            # RECIM simulator & risk manager
│   └── visualization/                      # Publication plotter (Figs 1-15) & diagram generator
│
├── configs/                                # YAML experiment configs
├── results/
│   ├── figures/                            # 300 DPI publication figures (Figs 5, 6, 7, 10, 11, 12, 13, 14)
│   ├── diagrams/                           # Research flowcharts (Figs 1, 4)
│   ├── tables/                             # CSV & Excel result summaries
│   └── predictions/                        # Test predictions per model & event
│
├── reports/
│   ├── final_research_report.md            # 30-Section academic research paper
│   ├── data_quality_report.md              # Cryptographic checksum & integrity report
│   └── feature_dictionary.csv              # Complete 84-feature dictionary
│
├── literature/                             # Literature benchmarks & gap analysis
│   ├── comparison_table.csv
│   └── related_work_analysis.md
│
└── tests/                                  # Full unit test suite (18 tests passing)
```
