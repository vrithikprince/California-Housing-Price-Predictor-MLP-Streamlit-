from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# 1. Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# 3. Define MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),   # two hidden layers: 64 & 32 neurons
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

# 4. Train
mlp.fit(X_train_scaled, y_train)

# 5. Validate/Test
print("Val R²:", mlp.score(X_val_scaled, y_val))
print("Test R²:", mlp.score(X_test_scaled, y_test))

# 6. Save model
joblib.dump(mlp, 'california_housing_mlp.pkl')