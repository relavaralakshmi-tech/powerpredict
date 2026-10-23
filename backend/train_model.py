"""Train a practical hourly electricity forecasting baseline from the supplied UCI-style text file."""
import os, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

load_dotenv()
DATA = os.getenv("DATASET_PATH", "data/household_power_consumption.txt")
MODEL = os.getenv("MODEL_PATH", "models/power_model.joblib")
os.makedirs(os.path.dirname(MODEL) or ".", exist_ok=True)

def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"Dataset not found: {DATA}. Put household_power_consumption.txt in the data/ folder.")
    # Read only the useful columns in chunks; aggregate to hourly means to keep training manageable.
    chunks = []
    for chunk in pd.read_csv(DATA, sep=";", na_values=["?"], usecols=["Date","Time","Global_active_power"],
                             chunksize=200_000, low_memory=False):
        chunk["datetime"] = pd.to_datetime(chunk["Date"] + " " + chunk["Time"], dayfirst=True, errors="coerce")
        chunk["power"] = pd.to_numeric(chunk["Global_active_power"], errors="coerce")
        chunk = chunk.dropna(subset=["datetime","power"])
        if not chunk.empty:
            hourly = chunk.set_index("datetime")["power"].resample("h").mean()
            chunks.append(hourly)
    if not chunks:
        raise ValueError("No valid power rows were found in the dataset.")
    series = pd.concat(chunks).groupby(level=0).mean().sort_index().resample("h").mean().interpolate(limit=6).dropna()
    frame = pd.DataFrame({"target": series})
    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["dayofyear"] = frame.index.dayofyear
    frame["weekend"] = (frame["dayofweek"] >= 5).astype(int)
    for lag in (1, 24, 168):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    frame = frame.dropna()
    if len(frame) < 200:
        raise ValueError("Not enough valid rows to train a model.")
    X = frame.drop(columns="target")
    y = frame["target"]
    split = max(int(len(frame)*0.8), 1)
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[split:]
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=20, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test) if len(X_test) else model.predict(X_train.tail(100))
    truth = y_test if len(y_test) else y_train.tail(100)
    metrics = {"mae_kw": float(mean_absolute_error(truth, pred)),
               "rmse_kw": float(np.sqrt(mean_squared_error(truth, pred))),
               "r2": float(r2_score(truth, pred)) if len(truth)>1 else None,
               "rows": int(len(frame)), "features": list(X.columns),
               "note": "Time-ordered holdout; metrics are in kW (hourly mean Global_active_power)."}
    dump({"model": model, "features": list(X.columns), "last_values": frame["target"].tail(168).tolist()}, MODEL)
    with open("models/model_metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL}")

if __name__ == "__main__":
    main()
