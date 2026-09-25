# Mid-Price Movement

The discriminative financial test evaluates whether models can classify real-time market price direction across transactions of limit order book dataset consisting market trade transactions over a full month, while continuously learning under a prequential (test-then-train) setting.

---

## About

Looking at the order flow feature set stream, predict whether the market mid-price will move up (+1) or down (-1) on immediate future horizons (1, 2, 3, 5, 10, 20, 50, 100, 200, and 500 trades ahead). To focus the evaluation on informative directional movements, models are evaluated only on non-zero future mid-price movement trade ticks.

**Challenge**
Unlike traditional machine learning test environment, financial markets constantly change. Models here must learn continually from live market data stream without stopping to retrain from scratch and output predicted classes. Once the future trade outcome occurs, the prediction is scored, and models are subjected to do inference and internal weight update from the observed outcome before the next trade arrives. This is referred to as prequential (test-then-train) environment.

Due to simulated continual nature of models (models are supposed to progressively see and evaluate new datapoints in a sequence), the test evaluates scalp to short term data, where prices fluctuate in high-frequency low signal noisy environment.

---

## Dataset

Bitcoin perpetual futures contract dataset containing high frequency market Level 4 order book data containing 11,918,929 transactions from December 1st to December 31st 2025, from Hyperliquid decentralized perpetual futures exchange.

>Albers, J., Cucuringu, M., Howison, S., & Shestopaloff, A. Y. (2026). An Open Book: Level 4 Order Book Data from the Hyperliquid Exchange (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.18184441

Analysis and data processing scripts are available in [`data/`](data/)

---

## Models

The test evaluates the **Sticker** at 80 standalone parameter variations, against 9 classical, statistical, and neural network baselines in continual learning environment.

**Neural Networks**
DeepLOB (CNN-LSTMs) Sequential convolution and deep sequence long-short term memory hybrid for limit order book,
Mamba (State-Space) State-Space S6 architecture,
TLOB (Transformer) Transformer with dual attention for limit order book

**Classical Statistical Baselines**
Logistic Regression, Kalman Filter, ARIMA, OFI Linear Regression, GARCH, LightGBM

Detailed per model configuration, metrics and runtime behaviour are available in [`sticker/`](sticker/) and [`baselines/`](baselines/)

Full predictions parquet files can be accessed from [`huggingface.co/aritraneogi/sticker/tree/main/mid_price_movement`](https://huggingface.co/aritraneogi/sticker/tree/main/mid_price_movement)

Baselines have weights and implementation scripts included.

---

## Comparison

Comparison of all 89 models are done on core model performance and trading viability across various scenarios which include continual regime adaptation, latency arbitrage and queue jumping and theoretical market taking and market making, derived from core model performance metrics.

The details are available in [`comparison/`](comparison/) in respective scenario folders.

---

For complete mathematical definitions, formal proofs, feature engineering equations, and baseline update rules, see [`MATH.md`](MATH.md)