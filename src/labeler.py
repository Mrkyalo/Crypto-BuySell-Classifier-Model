**Step 4 -Label Generation (Target Variable)**
df["future_return"] = df["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1

df["label"] = df.apply(label, axis=1)
def save_processed_csv(df, symbol, interval):

#Save a DataFrame into data/raw/(symbol_interval).csv
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_processed_csv(df, "BTCUSDT", "1dmodified")