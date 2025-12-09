import os
import requests
import pandas as pd
def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","num_trades",
                  "taker_base_volume","taker_quote_volume","ignore"]
    return df
df = fetch_binance("BTCUSDT", "1d", 1000)
print(df.head())
print(df.info())
def save_raw_csv(df, symbol="BTCUSDT", interval="1d"):
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

if __name__ == "__main__":
    df = fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000)
    save_raw_csv(df)
**Step 2 -Data Cleaning & Basic Processing**
df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
df["close_time"]=pd.to_datetime(df["close_time"],unit='ms')
df["close"] = df["close"].astype(float)
# convert the columns to float
df["close"] = df["close"].astype(float)
df["num_trades"] = df["num_trades"].astype(float)
df["open"] = df["open"].astype(float)
df["high"] = df["high"].astype(float)
df["low"] = df["low"].astype(float)
df["volume"] = df["volume"].astype(float)
df["quote_asset_volume"] = df["quote_asset_volume"].astype(float)
df["taker_base_volume"] = df["taker_base_volume"].astype(float)
df["taker_quote_volume"] = df["taker_quote_volume"].astype(float)
# Save processed data

def save_processed_csv(df, symbol, interval):
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_processed_csv(df, "BTCUSDT", "1d")
