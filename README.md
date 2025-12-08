## Crypto-BuySell-Classifier-ML Model
This is my Data science machine learning project at LuxDev.
The goal is to create a fully functioning ML system that:
-   Fetches real historical crypto data
-   Cleans and engineers features
-   Calculates technical indicators
-   Generates labels for Buy/Sell/Hold
-   Trains a classification model
-   Evaluates performance and backtests strategy
-   Serializes the trained model
-   Deploys prediction logic 

This is a realistic applied-finance ML workflow similar to what quant
researchers build.
## Features & Data
The model uses the following features derived from historical price and technical indicators:

- OHLCV: `open`, `high`, `low`, `close`, `volume`
- Quote & trade metrics: `quote_asset_volume`, `num_trades`, `taker_base_volume`, `taker_quote_volume`
- Returns & volatility: `return_1d`, `return_7d`, `volatility_7d`, `future_return`
- Technical indicators: `rsi`, `macd`, `sma_20`, `sma_50`, `sma_200`
- Bollinger Bands: `bb_high`, `bb_low`, `bb_pct`
- Stochastic RSI: `stoch_rsi`, `stoch_rsi_d`, `stoch_rsi_k`

The processed dataset is available in `data/processed/BTCUSDT_1dmodified.csv`.


