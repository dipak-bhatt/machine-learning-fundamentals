# Day 23 — Deploying a Machine Learning Model

**Module:** Module 6: Modern ML and Project

A model that cannot be called is not a product. Deployment turns a saved pipeline into
a service that other software can use reliably. That means defining an input and output
contract, validating requests, loading the model once, returning clear errors, and
monitoring the result after launch.

This day wraps the house-price pipeline from Day 21 in a FastAPI service. It covers
batch versus real-time serving, serialization, training-serving consistency, latency,
drift, retraining, security, and privacy. The API code is complete; FastAPI and
Uvicorn must be installed to run it.

## What You Will Learn

- How batch and real-time prediction differ.
- What a prediction API contract contains.
- How to serve a scikit-learn pipeline with FastAPI.
- How to validate input with Pydantic.
- How to load a model once at startup.
- How to expose health, model-info, single, and batch endpoints.
- How to test an API with Swagger UI, `curl`, and Python.
- How status codes communicate success and failure.
- How to prevent training-serving skew.
- How to containerize the service.
- What to monitor and when to retrain.
- How security and privacy apply to a prediction service.

## Detailed Content

- [Theory](theory.md) — notebook versus service, batch and real-time serving, HTTP
  and JSON, API contracts, serialization and version compatibility,
  training-serving consistency, latency and throughput, drift and monitoring,
  retraining, security and privacy, deployment patterns, and common mistakes.
- [Practical](practical.md) — train and save the pipeline, write a FastAPI app with
  Pydantic schemas and a lifespan loader, expose `/health`, `/model-info`, `/predict`,
  and `/predict-batch`, run it with Uvicorn, test with Swagger and `curl`, handle
  validation errors and unseen categories, sketch a container, and build a monitoring
  checklist.

## Files

- `train_model.py` — creates `house_price_pipeline.joblib` from the Day 4 CSV.
- `app.py` — the FastAPI service.
- `house_price_pipeline.joblib` — created when you run `train_model.py` or copied from
  Day 21.
- `requirements.txt` — serving dependencies.
- `Dockerfile` — optional container recipe.

Requires `fastapi` and `uvicorn`, which are not installed in the base environment:

```bash
pip install fastapi "uvicorn[standard]"
```

No data download is required.

## Before You Start

- Day 21: the complete house-price pipeline and `joblib`.
- Day 20: pipelines and preprocessing.
- Day 05: train/test discipline and baseline evaluation.
- Basic familiarity with HTTP and JSON is helpful but not assumed.

## Outcome

Learners can package a trained pipeline as a FastAPI service, define and validate a
request schema, expose single and batch prediction endpoints, handle errors with the
right status codes, test the service, containerize it, and describe the monitoring,
security, and retraining needed for responsible deployment.
