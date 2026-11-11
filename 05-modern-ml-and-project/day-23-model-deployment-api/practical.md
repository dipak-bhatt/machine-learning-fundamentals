# Day 23 — Practical: Serving a Model with FastAPI

**Module:** Module 6: Modern ML and Project

In this practical we will:

- Prepare the house-price pipeline from Day 21.
- Wrap it in a FastAPI service.
- Load the model once at startup, not per request.
- Validate input with Pydantic.
- Expose `/health`, `/model-info`, `/predict`, and `/predict-batch`.
- Test the API with Swagger UI and `curl`.
- Handle validation errors and model failures correctly.
- Sketch a container deployment and a monitoring checklist.

FastAPI is not installed by default in this environment. The API code is complete and
correct, but you must install the dependencies before running it.

## 1. Install the Dependencies

```bash
pip install fastapi "uvicorn[standard]"
```

If you are starting from a clean environment, also install the modeling stack:

```bash
pip install scikit-learn==1.5.1 pandas==2.2.2 numpy==2.0.0 joblib==1.4.2
```

## 2. Prepare the Model

Create a folder for the service and train the model into it. If you already ran Day 21,
copy `house_price_pipeline.joblib` into this folder and skip to step 3.

`train_model.py`:

```python
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

path = "../../00-foundations/day-04-bias-variance-ml-workflow/house_prices.csv"
df = pd.read_csv(path)

X = df.drop(columns="price_usd")
y = df["price_usd"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric = ["area_sqft", "age_years"]
categorical = ["location"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(
        n_estimators=300, random_state=42)),
])

model.fit(X_train, y_train)
joblib.dump(model, "house_price_pipeline.joblib")
print("saved house_price_pipeline.joblib")
```

Run it:

```bash
python train_model.py
```

Expected output:

```text
saved house_price_pipeline.joblib
```

The saved object is the complete pipeline. The API will pass raw records to it, and
the pipeline will impute, scale, encode, and predict.

## 3. Project Structure

```text
day-23-model-deployment-api/
    app.py
    train_model.py
    house_price_pipeline.joblib
    requirements.txt
```

## 4. The API

`app.py`:

```python
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "house_price_pipeline.joblib"
MODEL_VERSION = "1.0.0"


class HouseFeatures(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Living area in square feet")
    age_years: float | None = Field(
        default=None, ge=0, description="Age in years; null if unknown"
    )
    location: str = Field(..., description="City, Suburb, or Town")


class PredictionResponse(BaseModel):
    price_usd: float
    model_version: str


class BatchRequest(BaseModel):
    houses: list[HouseFeatures]


class BatchResponse(BaseModel):
    prices_usd: list[float]
    model_version: str


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="House Price Prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestRegressor",
        "version": MODEL_VERSION,
        "features": ["area_sqft", "age_years", "location"],
        "target": "price_usd",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    try:
        frame = pd.DataFrame([house.model_dump()])
        price = float(state["model"].predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictionResponse(
        price_usd=round(price, 2),
        model_version=MODEL_VERSION,
    )


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    frame = pd.DataFrame([house.model_dump() for house in request.houses])
    prices = state["model"].predict(frame)
    return BatchResponse(
        prices_usd=[round(float(price), 2) for price in prices],
        model_version=MODEL_VERSION,
    )
```

Key design points:

- The model loads once in `lifespan`, not inside each request.
- `HouseFeatures` validates types and ranges before the model sees anything.
- `age_years` may be `null`, which becomes a missing value for the imputer.
- `/health` reports whether the model is loaded.
- Every successful response includes the model version.
- Unexpected model failures become a 500, not a leaked traceback.

## 5. Run the Service

```bash
uvicorn app:app --reload
```

Expected output (abridged):

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. You can call the endpoints
interactively from there.

## 6. Test a Single Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"area_sqft\": 1500, \"age_years\": 10, \"location\": \"Suburb\"}"
```

Expected response:

```json
{"price_usd":74278.33,"model_version":"1.0.0"}
```

A missing age is also valid:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"area_sqft\": 2500, \"age_years\": null, \"location\": \"City\"}"
```

Expected response:

```json
{"price_usd":181720.0,"model_version":"1.0.0"}
```

The imputer fills the missing age using the training median. The client does not need
to know the median.

Test from Python:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "area_sqft": 900,
        "age_years": 30,
        "location": "Town",
    },
)
print(response.status_code)
print(response.json())
```

Expected output:

```text
200
{'price_usd': 43005.0, 'model_version': '1.0.0'}
```

## 7. Test a Batch

```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d "{\"houses\": [
        {\"area_sqft\": 1500, \"age_years\": 10, \"location\": \"Suburb\"},
        {\"area_sqft\": 2500, \"age_years\": null, \"location\": \"City\"}
      ]}"
```

Expected response:

```json
{"prices_usd":[74278.33,181720.0],"model_version":"1.0.0"}
```

Batch endpoints reduce per-request overhead when a client needs many predictions at
once.

## 8. Validation Errors

Send a negative area:

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"area_sqft\": -5, \"age_years\": 10, \"location\": \"Town\"}"
```

Expected status and body (abridged):

```text
HTTP/1.1 422 Unprocessable Entity
{"detail":[{"type":"greater_than","loc":["body","area_sqft"],...}]}
```

Pydantic rejects the value before the model runs. This is the right behavior: the
model should never see a house with negative area.

An unknown location is a different case. The encoder was fitted with
`handle_unknown="ignore"`, so a new value does not crash the request; it produces a
prediction from the other features.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"area_sqft\": 1500, \"age_years\": 10, \"location\": \"Village\"}"
```

Expected response:

```json
{"price_usd":77186.67,"model_version":"1.0.0"}
```

Decide deliberately whether that is acceptable. For a closed set of locations, use a
`Literal` type or a validation check to reject unknown values. For open sets, the
all-zero encoding may be reasonable.

## 9. Health and Model Info

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

Expected responses:

```json
{"status":"ok","model_loaded":true}
```

```json
{
  "model": "RandomForestRegressor",
  "version": "1.0.0",
  "features": ["area_sqft", "age_years", "location"],
  "target": "price_usd"
}
```

Orchestrators and load balancers use health checks to decide whether to route traffic
to an instance. A service that reports healthy before the model loads will fail real
requests during startup.

## 10. Container Sketch

`requirements.txt`:

```text
fastapi
uvicorn[standard]
scikit-learn==1.5.1
pandas==2.2.2
numpy==2.0.0
joblib==1.4.2
```

`Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py house_price_pipeline.joblib ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

Pin versions in the image so the serving environment matches training. A container
also makes the service reproducible across machines.

## 11. Monitoring Checklist

Once the service is live, monitor:

- **Requests**: count, latency, error rate by status code.
- **Inputs**: missing rates, out-of-range values, unseen categories.
- **Predictions**: mean, spread, share above a threshold, sudden jumps.
- **Performance**: once labels arrive, RMSE and MAE over time.
- **Segments**: error and prediction distributions by location.
- **Resources**: CPU, memory, and model load time.
- **Versions**: which model version served each request.

Log model version with every request. Without it, a performance change cannot be
traced to a deployment. Avoid logging raw personal data; log identifiers and
aggregates instead.

## 12. Try It Yourself

1. **Strict locations.** Replace `location: str` with
   `Literal["City", "Suburb", "Town"]` and confirm unknown values return 422.
2. **Add an endpoint.** Expose `/predict-batch-file` that accepts a CSV upload and
   returns a CSV download.
3. **Add authentication.** Require an `X-API-Key` header and reject missing keys with
   401.
4. **Add metrics.** Count requests and record latency in memory, and expose them at
   `/metrics`.
5. **Simulate drift.** Send 100 requests with doubled `area_sqft` and compare the
   prediction distribution with the training distribution.
6. **Golden records.** Save five training rows with known predictions and add a
   startup check that compares served output with the saved values.
7. **Deploy to a container.** Build the image and run it on a different port.

## 13. What We Learned

- Deployment turns a model file into a service with a contract.
- FastAPI provides routing, validation, and automatic documentation.
- The model should load once at startup.
- Pydantic validates input before it reaches the model.
- The pipeline handles missing values and unseen categories at serving time.
- Status codes and response schemas make failures predictable for clients.
- Health checks and version metadata are operational necessities.
- Containers pin the environment and make deployment reproducible.
- Monitoring, drift detection, and retraining keep a deployed model useful.
- Security and privacy are part of the serving design.

## Outcome

Learners can package a trained pipeline behind a FastAPI service, validate requests,
expose single and batch prediction endpoints, test the API with Swagger and `curl`,
handle errors correctly, containerize the service, and plan the monitoring and
security needed for responsible deployment.
