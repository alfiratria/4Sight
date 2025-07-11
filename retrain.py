# model/retrain.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv("model/training_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# 2. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Training
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# 4. Simpan model dan scaler
joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")
