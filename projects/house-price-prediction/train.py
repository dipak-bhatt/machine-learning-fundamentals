import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("R2:", r2_score(y_test, pred))
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/house_price_ridge.joblib")
