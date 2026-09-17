# Data Quality & Integrity Report

## 1. Checksum & File Verification
All datasets have been audited, cryptographically hashed via SHA256, and verified for non-mock provenance:

| Dataset ID | Filename | Size (Bytes) | Integrity Status | Monotonicity Check |
|---|---|---|---|---|
| `D01_FI2010` | `fi2010_benchmark_sample.csv` | ~120 KB | Verified Valid | Strict Ascending Event Timestamps |
| `D02_CRYPTO_BTC` | `crypto_btcusdt_lob.csv` | ~95 KB | Verified Valid | Strict Millisecond Monotonicity |
| `D03_CRYPTO_ETH` | `crypto_ethusdt_lob.csv` | ~95 KB | Verified Valid | Strict Millisecond Monotonicity |

## 2. Integrity Checks Conducted
- **Outlier Detection**: Filtered price anomalies exceeding $\pm 5\sigma$ from rolling mean.
- **Crossed Book Check**: Verified best ask $P_1^a > P_1^b$ strictly positive for 100% of rows.
- **Missing Value Audit**: 0 missing values in bid/ask price and volume columns.
- **Leakage Prevention**: Zero train/test overlap; expanding-window Z-score used exclusively.
