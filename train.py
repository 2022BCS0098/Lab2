import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ PATHS ------------------
DATA_PATH = "winequality-red.csv"
OUTPUT_DIR = "outputs/linear"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# ------------------ TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ PIPELINE ------------------
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# ------------------ TRAIN ------------------
pipeline.fit(X_train, y_train)

# ------------------ EVALUATE ------------------
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ OUTPUT ------------------
print(f"Linear Regression MSE: {mse}")
print(f"Linear Regression R2: {r2}")

joblib.dump(pipeline, MODEL_PATH)

with open(RESULTS_PATH, "w") as f:
    json.dump({"mse": mse, "r2_score": r2}, f, indent=4)
