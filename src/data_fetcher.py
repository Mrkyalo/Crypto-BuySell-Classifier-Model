import os
import requests
import pandas as pd

#Fetch data on Binance API
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

#show data
df = fetch_binance("BTCUSDT", "1d", 1000)
print(df.head())
print(df.info())

#save raw data

def save_raw_csv(df, symbol="BTCUSDT", interval="1d"):
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{symbol}_{interval}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

if __name__ == "__main__":
    df = fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000)
    save_raw_csv(df)
