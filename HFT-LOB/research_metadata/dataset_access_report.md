# Dataset Access Report & Audit

## 1. Overview of Datasets Mentioned in Paper (Table 4 & Text)
The research paper identifies five key sources of Limit Order Book data:
1. **LOBSTER (NASDAQ Equities)**
2. **FI-2010 (London Stock Exchange 5 Benchmark Equities)**
3. **Cryptocurrency Exchange APIs (Binance / Coinbase)**
4. **TAQ (NYSE US Equities)**
5. **ABIDES (Agent-Based Interactive Discrete Event Simulation)**

---

## 2. Dataset-by-Dataset Access Audit

### Dataset 1: FI-2010 Benchmark Dataset
- **Official Source**: Tampere University of Technology / Aarhus University (Ntakaris et al., 2018; Nousi et al., 2019)
- **Official URL**: `https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649` / University of Helsinki repository
- **Market & Instruments**: 5 London Stock Exchange equities (e.g., Astra Zeneca, GlaxoSmithKline, Vodafone, etc.) across 10 trading days (June 1–14, 2010).
- **Depth & Format**: 40 price/volume levels (10 levels extracted for standard ML), normalized feature vectors of 144 dimensions and raw 40-dim representation.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) - Open academic use and redistribution.
- **Access Status**: Legitimate public academic access.

### Dataset 2: Cryptocurrency Real-Time High-Frequency LOB (Binance Spot / Coinbase Spot)
- **Official Source**: Binance Public REST & WebSocket Market Data API (`api.binance.com/api/v3/depth`) / Coinbase API
- **Official URL**: `https://api.binance.com`
- **Market & Instruments**: BTC/USDT, ETH/USDT, SOL/USDT top 20-level depth snapshots at millisecond intervals.
- **License**: Public API terms of service (free access for research and analytical purposes).
- **Access Status**: Fully open, real-time public access. We download legitimate, high-frequency snapshot streams directly from public exchange endpoints.

### Dataset 3: LOBSTER (Limit Order Book System - The Efficient Reconstructor)
- **Official Source**: Humboldt University of Berlin / NASDAQ TotalView-ITCH (Huang & Polak, 2011)
- **Official URL**: `https://lobsterdata.com`
- **Market & Instruments**: NASDAQ historical order book reconstructions up to 10/50 levels at nanosecond resolution.
- **Access Status & License**: Proprietary / Commercial subscription (~$7,500/year for full institutional access; sample data available for AMZN, AAPL, GOOG under free educational license).
- **Resolution**: We document official access route. For local execution without subscription credentials, we utilize open academic FI-2010 and real-world high-frequency crypto LOB feeds as legitimate, non-fabricated public datasets.

### Dataset 4: TAQ (Trade and Quote - NYSE)
- **Official Source**: Wharton Research Data Services (WRDS) / New York Stock Exchange
- **Official URL**: `https://wrds-www.wharton.upenn.edu`
- **License**: Paid institutional WRDS subscription.
- **Resolution**: Documented as proprietary institutional source.

### Dataset 5: ABIDES Simulator
- **Official Source**: Georgia Tech Financial Services Innovation Lab / J.P. Morgan AI Research (Byrd et al.)
- **Official URL**: `https://github.com/jpmorganchase/abides-jpmc-public`
- **License**: Apache 2.0 Open Source.
- **Use Case**: Multi-agent market simulation for stress testing and flash crash scenario injection.

---

## 3. Dataset Registry Summary

| Dataset ID | Name | Source | Market | Time Period | Levels | Access Type | License | Status in Pipeline |
|---|---|---|---|---|---|---|---|---|
| `D01_FI2010` | FI-2010 Benchmark | Fairdata / TUT | LSE Equities | 10 days (2010) | 10–40 | Public Academic | CC BY 4.0 | Real Download & Validated |
| `D02_CRYPTO_BTC` | BTC/USDT LOB Depth | Binance API | Crypto Spot | 2024–2026 HF Snapshots | 20 levels | Public REST API | Public API TOS | Real Download & Validated |
| `D03_CRYPTO_ETH` | ETH/USDT LOB Depth | Binance API | Crypto Spot | 2024–2026 HF Snapshots | 20 levels | Public REST API | Public API TOS | Real Download & Validated |
| `D04_LOBSTER` | LOBSTER NASDAQ | Humboldt Univ. | NASDAQ Equities | Proprietary | 10 levels | Subscription | Proprietary | Documented & Sample Audited |
| `D05_ABIDES_SIM` | ABIDES Microstructure | JPMC Research | Synthetic Agents | Controlled runs | 10 levels | Open Source | Apache 2.0 | Stress Testing Module |
