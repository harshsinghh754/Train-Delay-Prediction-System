# train_ml_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle

DATA_PATH = "train_delay_data.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Rows loaded:", len(df))

# --- Clean basic data ---

df = df.dropna(subset=["delay_minutes"])
df["delay_minutes"] = df["delay_minutes"].astype(int)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.weekday  # 0 = Monday, 6 = Sunday

DELAY_THRESHOLD = 15
df["is_delayed"] = (df["delay_minutes"] > DELAY_THRESHOLD).astype(int)

# --- Encode weather text -> number ---

weather_encoder = LabelEncoder()
df["weather_encoded"] = weather_encoder.fit_transform(df["weather"].astype(str))

# --- Features & targets ---

feature_cols = ["distance_km", "temp_c", "weather_encoded", "month", "day_of_week"]

for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in CSV: {col}")

X = df[feature_cols].values
y_delay = df["delay_minutes"].values
y_is_delayed = df["is_delayed"].values

# --- Split train/test ---

X_train, X_test, y_delay_train, y_delay_test, y_cls_train, y_cls_test = train_test_split(
    X, y_delay, y_is_delayed, test_size=0.2, random_state=42
)

# --- Model 1: Delay minutes (regression) ---

print("\nTraining RandomForestRegressor (delay minutes)...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_delay_train)

y_delay_pred = reg_model.predict(X_test)
mae = mean_absolute_error(y_delay_test, y_delay_pred)
print(f"Mean Absolute Error: {mae:.2f} minutes")

# --- Model 2: Delayed or not (classification) ---

print("\nTraining RandomForestClassifier (delayed yes/no)...")
cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
cls_model.fit(X_train, y_cls_train)

y_cls_pred = cls_model.predict(X_test)
acc = accuracy_score(y_cls_test, y_cls_pred)
print(f"Classification Accuracy: {acc * 100:.2f}%")

# --- Save models & encoder ---

print("\nSaving models...")
with open("delay_regression_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)

with open("delay_classification_model.pkl", "wb") as f:
    pickle.dump(cls_model, f)

with open("weather_encoder.pkl", "wb") as f:
    pickle.dump(weather_encoder, f)

print("Models saved successfully as:")
print("  delay_regression_model.pkl")
print("  delay_classification_model.pkl")
print("  weather_encoder.pkl")

# --- Demo prediction ---

def demo():
    distance_km = 800
    temp_c = 25
    weather_text = "Fog"   # try "Clear", "Rain", etc.
    month = 1
    day_of_week = 4  # Friday

    weather_encoded = weather_encoder.transform([weather_text])[0]
    features = np.array([[distance_km, temp_c, weather_encoded, month, day_of_week]])

    pred_delay = reg_model.predict(features)[0]
    prob_delay = cls_model.predict_proba(features)[0][1]
    delayed_label = cls_model.predict(features)[0]

    print("\n--- Demo Prediction ---")
    print(f"Predicted delay (minutes): {pred_delay:.2f}")
    print(f"Chance of delay: {prob_delay * 100:.2f}%")
    print(f"Will it be delayed (> {DELAY_THRESHOLD} min)? {delayed_label}")

if __name__ == "__main__":
    demo()
