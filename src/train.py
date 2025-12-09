**Step 5 - Train/Test Split**
import sys
print(sys.executable)
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,classification_report
df=pd.read_csv(r"C:\Users\HomePC\Crypto-BuySell-Classifier-Model\notebooks\data\processed\BTCUSDT_1d_features.csv")
df.head()
print(df.columns.tolist())

# --- Create target column ---

# future return
df["future_return"] = df["close"].shift(-1) / df["close"] - 1

# Buy / Hold / Sell labels
df["label"] = df["future_return"].apply(
    lambda x: 1 if x > 0.003 else (-1 if x < -0.003 else 0)
)

# drop last row (no future close)
df = df.dropna(subset=["future_return"]).reset_index(drop=True)


# --- Define features + target ---

target_col = "label"
feature_cols = df.columns.difference(["label", "future_return", "open_time", "close_time"])

X = df[feature_cols]
y = df[target_col]


# --- Train / Val / Test split ---

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_start = train_size + val_size

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:test_start]
y_val = y.iloc[train_size:test_start]

X_test = X.iloc[test_start:]
y_test = y.iloc[test_start:]


# --- Print summary ---
print(f"Train: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")
*** Step 6 -- Model Training
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
df = pd.read_csv(r"C:\Users\HomePC\Crypto-BuySell-Classifier-Model\notebooks\data\processed\BTCUSDT_1dmodified.csv")
#Define features and target variable
feature_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'num_trades',
    'taker_base_volume', 'taker_quote_volume',
    'return_1d', 'return_7d', 'volatility_7d',
    'rsi', 'macd', 'sma_20', 'sma_50', 'sma_200',
    'bb_high', 'bb_low', 'bb_pct',
    'stoch_rsi', 'stoch_rsi_d', 'stoch_rsi_k',
    'future_return'
]

X = df[feature_cols]
y = df["label"]
#Fix missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
#Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)
#Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
***Logistic Regression
#Logistic Regression Model

model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced',max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

*** Random forest classifier
# Random Forest Classifier
model_rf = RandomForestClassifier(class_weight="balanced")
model_rf.fit(X_train_scaled, y_train)

*** Catboost model
# CatBoost model
model_cat = CatBoostClassifier(
    depth=6,
    learning_rate=0.03,
    loss_function="MultiClass",
    verbose=0
)

model_cat.fit(X_train, y_train)

*** LightGBM Model
#LightGBM model
model_lgb = LGBMClassifier(class_weight="balanced")
# --- Train ---
model_lgb.fit(X_train, y_train)
*** XGBoost model
#XGBoost model
model_xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss"
)
model_xgb.fit(X_train, y_train)
*** Step 7-Evalution of the models
#Evaluation of logistic regression
y_pred_lr = model_lr.predict(X_val_scaled)
print("\n=== Logistic Regression ===")
print(classification_report(y_val, y_pred_lr, digits=4))
print(confusion_matrix(y_val, y_pred_lr))
# Catboost Evaluation
y_pred_cat = model_cat.predict(X_val)
print("\n=== CatBoost ===")
print(classification_report(y_val, y_pred_cat, digits=4))
print(confusion_matrix(y_val, y_pred_cat))
# Evaluation of LightGBM
y_pred_lgb = model_lgb.predict(X_val)
print("\n=== LightGBM ===")
print(classification_report(y_val, y_pred_lgb, digits=4))
print(confusion_matrix(y_val, y_pred_lgb))
# Evluation random Forest
y_pred_rf = model_rf.predict(X_val)
print("\n=== Random Forest ===")
print(classification_report(y_val, y_pred_rf, digits=4))
print(confusion_matrix(y_val, y_pred_rf))
#XGBoost Evaluation
y_val_pred = model_xgb.predict(X_val)
print("\n=== XGBoost ===")
print(classification_report(y_val, y_val_pred, digits=4))
print(confusion_matrix(y_val, y_val_pred))

# Backtesting
# Initial capital
capital = 10000
prices = df["close"].values[-len(y_val):]  # validation period prices
signals = model_lr.predict(X_val_scaled) # replace with any model predictions: 0=HOLD, 1=BUY, 2=SELL

# Simulate strategy
def simulate(prices, signals):
    cash, coin = capital, 0
    portfolio = []
    for p, s in zip(prices, signals):
        if s == 1 and cash > 0:    # BUY
            coin, cash = cash / p, 0
        elif s == 2 and coin > 0:  # SELL
            cash, coin = coin * p, 0
        portfolio.append(cash + coin * p)
    return portfolio

model_portfolio = simulate(prices, signals)
buy_hold_portfolio = [capital / prices[0] * p for p in prices]
random_portfolio = simulate(prices, np.random.choice([0,1,2], len(prices)))

# Plot results
plt.figure(figsize=(10,5))
plt.plot(model_portfolio, label="Model")
plt.plot(buy_hold_portfolio, label="Buy & Hold")
plt.plot(random_portfolio, label="Random")
plt.xlabel("Time")
plt.ylabel("Portfolio Value ($)")
plt.title("Trading Strategy Backtest")
plt.legend()
plt.show()

# Final results
print("Final portfolio values:")
print(f"Model: ${model_portfolio[-1]:.2f}")
print(f"Buy & Hold: ${buy_hold_portfolio[-1]:.2f}")
print(f"Random: ${random_portfolio[-1]:.2f}")
** Logistic regression is our best model to use
# Step 8-Serialize the Model
# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Export all trained models
joblib.dump(model_lr, "models/logistic_regression.pkl")
joblib.dump(model_rf, "models/random_forest.pkl")
joblib.dump(model_cat, "models/catboost.pkl")
joblib.dump(model_lgb, "models/lightgbm.pkl")
joblib.dump(model_xgb, "models/xgboost.pkl")

# Also save the scaler for preprocessing
joblib.dump(scaler, "models/scaler.pkl")

# Save feature columns for reference
import json
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_cols, f)

print("All models and preprocessing artifacts saved successfully!")
#Best model: Logistic Regression (save)

joblib.dump(model_lr, "models/best_model.pkl")
print("Best model (logistic_regression) saved as models/best_model.pkl")

