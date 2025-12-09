# Step 9-Prediction Pipeline
def predict(features):
    model = joblib.load("models/buy_sell_classifier.pkl")
    return model.predict(features)
#Predict function example
model = joblib.load("models/logistic_regression.pkl")

def predict(features):
    return model.predict(features)
features = [[40000, 40500, 39500, 40200, 1234,5555,
              1200, 600, 300,0.01, 0.03, 0.05,
              55,-12, 39700, 39000, 35000,
              41000, 39000, 0.6, 0.7, 0.65, 0.72, 0.02]]
print(predict(features))