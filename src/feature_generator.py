**Step 3 Feature Engineering**
import sys
print(sys.executable)
import pandas as pd
import ta
from pathlib import Path
import matplotlib.pyplot as plt
import os
df=pd.read_csv(r"C:\Users\HomePC\Crypto-BuySell-Classifier-Model\notebooks\data\processed\BTCUSDT_1d.csv")
df.info()
def add_basic_features(df):

    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_7d'] = df['close'].pct_change(7)
    
    # Rolling volatility
    df['volatility_7d'] = df['return_1d'].rolling(window=7).std()
    
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()
    
    # MACD
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    
    # Moving averages
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()
    
    # Stochastic RSI
    stoch = ta.momentum.StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    df["stoch_rsi"] = stoch.stochrsi()
    df["stoch_rsi_d"] = stoch.stochrsi_d()
    df["stoch_rsi_k"] = stoch.stochrsi_k()
    
    # Drop NaNs caused by rolling windows
    df = df.dropna().reset_index(drop=True)
    return df
def process_and_save(raw_csv_path, out_dir="data/processed"):
    """Load raw CSV, add features, and save processed CSV."""
    raw_csv_path = Path(raw_csv_path)
    df = pd.read_csv(raw_csv_path)
    
    # Compute all features
    df_features = add_basic_features(df)
    
    # Ensure output directory exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Save processed features
    out_path = Path(out_dir) / raw_csv_path.name.replace(".csv", "_features.csv")
    df_features.to_csv(out_path, index=False)
    print(f"Saved features to: {out_path}")
    
    return df_features
if __name__ == "__main__":
    raw_file = r"C:\Users\HomePC\Crypto-BuySell-Classifier-Model\notebooks\data\raw\BTCUSDT_1d.csv"
    df = process_and_save(raw_file)
    
    print("Processed DataFrame shape:", df.shape)
    print(df.head())


# After you run process_and_save and get df_features
df = process_and_save(raw_file)

# Create subplots
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

# 1. Closing price with SMA overlays
axes[0].plot(df["close"], label="Close", color="black")
axes[0].plot(df["sma_20"], label="SMA20", color="blue")
axes[0].plot(df["sma_50"], label="SMA50", color="orange")
axes[0].plot(df["sma_200"], label="SMA200", color="red")
axes[0].set_title("BTC Close with Moving Averages")
axes[0].legend()

# 2. RSI
axes[1].plot(df["rsi"], label="RSI(14)", color="purple")
axes[1].axhline(70, linestyle="--", color="red")
axes[1].axhline(30, linestyle="--", color="green")
axes[1].set_title("Relative Strength Index")
axes[1].legend()

# 3. MACD diff
axes[2].plot(df["macd"], label="MACD diff", color="brown")
axes[2].axhline(0, linestyle="--", color="black")
axes[2].set_title("MACD Difference")
axes[2].legend()

# 4. Bollinger Bands
axes[3].plot(df["close"], label="Close", color="black")
axes[3].plot(df["bb_high"], label="BB High", color="green")
axes[3].plot(df["bb_low"], label="BB Low", color="red")
axes[3].set_title("Bollinger Bands")
axes[3].legend()

# 5. Stochastic RSI
axes[4].plot(df["stoch_rsi"], label="StochRSI", color="blue")
axes[4].plot(df["stoch_rsi_d"], label="%D", color="orange")
axes[4].plot(df["stoch_rsi_k"], label="%K", color="green")
axes[4].axhline(0.8, linestyle="--", color="red")
axes[4].axhline(0.2, linestyle="--", color="green")
axes[4].set_title("Stochastic RSI")
axes[4].legend()

plt.tight_layout()
plt.show()
