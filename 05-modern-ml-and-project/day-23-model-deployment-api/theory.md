# Day 23 — Theory: Deploying a Machine Learning Model

**Module:** Module 6: Modern ML and Project

A model becomes useful when something else can call it. That "something else" might be
a web application, a mobile app, a batch job, or another service. Deployment is the
discipline of making the model available reliably, consistently, and safely.

This day covers the two main serving patterns, the anatomy of a prediction API, the
serialization and consistency problems that break deployments, and the operational
concerns — monitoring, drift, security, and privacy — that determine whether a model
stays useful after launch.

## In Plain English

**Think of it like this:** deployment turns a model sitting in a notebook into a
service other software can call over the network.

- Save the trained model (for example with `joblib`) and load it inside a small
  web service.
- An **API** exposes an endpoint such as `/predict` that accepts JSON and returns
  a prediction.
- **Validate inputs**, handle errors, and return clear responses.
- Monitor latency, errors, and **data drift** after release.
- Start simple — one endpoint, one model — before adding scale.

## 1. From Notebook to Service

A notebook runs once, interactively, with a human watching. A service runs repeatedly,
unattended, for many users, often under load. The differences matter:

- **Inputs are not curated.** Users send missing fields, wrong types, and unseen
  categories.
- **The process is long-lived.** It must start reliably and stay healthy.
- **Latency matters.** A response must arrive in milliseconds or seconds.
- **Failures are visible.** An exception becomes a failed request, not a red cell.
- **Data changes.** The distribution of inputs drifts over time.
- **Security matters.** The endpoint is reachable by others.

Deployment is therefore not a single step. It is a set of engineering decisions around
the model.

## 2. Batch versus Real-Time Prediction

There are two broad serving patterns.

### Batch prediction

Predictions are computed on a schedule for many records at once and stored.

```text
database snapshot -> batch job -> predictions table -> dashboard or downstream job
```

Use batch when:

- Predictions are needed periodically, not on demand.
- Latency is not critical.
- The input set is large and known in advance.
- You want to keep the serving system simple.

Examples: nightly churn scores, weekly demand forecasts, daily risk tiers.

### Real-time prediction

Predictions are computed on request, one or a few records at a time.

```text
client -> API -> model -> response
```

Use real-time when:

- A user is waiting for the result.
- The input depends on the current interaction.
- Predictions must reflect the latest data.

Examples: fraud checks during checkout, price estimates on a listing page, spam
classification as a message arrives.

### Hybrid

Many systems combine both: a batch job precomputes heavy features or default scores,
and a real-time endpoint adjusts them for the current request. This is common in
recommendation and ranking systems.

## 3. What an API Is

An **application programming interface (API)** is a contract for how software talks to
software. A web API exchanges messages over HTTP:

- The client sends a **request**: method, URL, headers, body.
- The server returns a **response**: status code, headers, body.

For prediction APIs, the common pattern is:

- `POST /predict` with a JSON body of features.
- A JSON response containing the prediction.

```http
POST /predict HTTP/1.1
Content-Type: application/json

{"area_sqft": 1500, "age_years": 10, "location": "Suburb"}
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{"price_usd": 74278.33}
```

Status codes communicate what happened:

- **200**: success.
- **400**: malformed request.
- **422**: valid JSON but invalid values (validation error).
- **500**: server error, including model failure.

Using the right status code lets clients react correctly instead of parsing error
messages.

## 4. Designing the Prediction Endpoint

A good endpoint is explicit about its contract:

- **Input schema**: field names, types, allowed ranges, optional fields.
- **Output schema**: field names and units.
- **Version**: which model or API version produced the result.
- **Errors**: what invalid input produces.

Example contract:

```text
POST /predict
input:
  area_sqft: number > 0
  age_years: number >= 0 or null
  location: string
output:
  price_usd: number
  model_version: string
```

Validation belongs at the boundary. Reject bad input early with a clear message rather
than letting it reach the model and produce a nonsensical prediction.

## 5. Serialization and Version Compatibility

A trained model must be stored in a form the serving process can load:

- **joblib / pickle**: common for scikit-learn. Fast for NumPy arrays.
- **ONNX**: a portable format for cross-framework inference.
- **TorchScript / SavedModel**: framework-specific formats for deep learning.
- **Custom formats**: JSON weights for simple models.

Compatibility risks:

- A model saved with scikit-learn 1.4 may not load correctly in 1.5.
- Python version differences can break pickle loading.
- Custom classes in the pipeline must be importable in the serving environment.

Mitigations:

- Pin library versions in both training and serving.
- Save the **entire pipeline**, including preprocessing.
- Test loading in a clean environment that matches production.
- Keep the training environment reproducible.

A model file is not self-describing. Store metadata: training date, data version,
metrics, package versions, and intended use.

## 6. Training-Serving Consistency

The most common deployment bug is **training-serving skew**: the features the model
sees in production differ from the features it saw in training.

Causes:

- Preprocessing reimplemented in the serving code and drifting from training.
- Different default values for missing fields.
- Different units (dollars vs cents, metres vs feet).
- Different category mappings.
- Feature computed from data available at training but not at serving time.

The defense is to make the pipeline the single source of truth. Save the fitted
pipeline; load it and call `predict` on raw input. Do not rebuild preprocessing by
hand.

Check consistency with a small set of **golden records**: inputs with known outputs
from training. If the served model disagrees, stop the deployment.

## 7. Latency and Throughput

- **Latency**: time to serve one request.
- **Throughput**: requests served per second.

Factors:

- Model size and complexity.
- Feature computation.
- Serialization and network overhead.
- Concurrency and hardware.

Techniques:

- Load the model **once at startup**, not per request.
- Cache expensive features.
- Batch requests when possible.
- Use efficient serialization.
- Scale horizontally with multiple workers.
- Use a GPU only if the model benefits enough to justify it.

A model that takes 30 seconds per prediction may be accurate and still unusable for an
interactive product.

## 8. Model Drift and Monitoring

Deployed models degrade. Monitoring detects it.

### Data drift

The distribution of inputs changes. Example: a new region appears, or average house
size shifts. The model may still run but its assumptions no longer hold.

### Concept drift

The relationship between inputs and target changes. Example: a market shock changes
how buyers value location.

### Pipeline drift

An upstream data source changes format, units, or meaning. Example: a field that used
to be in dollars is now in cents.

What to monitor:

- Input distributions: means, ranges, missing rates, category frequencies.
- Prediction distributions: average score, share of positives, outliers.
- Live performance: once labels arrive, compute the same metrics as in training.
- Errors and latency: 4xx, 5xx, timeouts.
- Segment performance: drift can hit one group first.

Monitoring needs a reference: the training distribution and the last known good
period. Alert on meaningful deviations, not every fluctuation.

## 9. Retraining

When performance degrades, retrain. Decide:

- **Trigger**: schedule, drift threshold, or performance drop.
- **Data window**: all history, recent window, or weighted recent data.
- **Validation**: the same honest protocol as before.
- **Rollout**: shadow mode, canary, or full replacement.
- **Rollback**: keep the previous version ready.

A retrained model is a new model. It needs the same evaluation, and it should be
versioned and documented.

## 10. Security and Privacy

A public endpoint is an attack surface.

- **Authentication**: require an API key or token.
- **Authorization**: limit who can call what.
- **Rate limiting**: prevent abuse and accidental overload.
- **Input size limits**: reject huge payloads.
- **No secrets in code or logs**: keys, credentials, personal data.
- **TLS**: encrypt traffic.
- **Dependency hygiene**: pin and patch libraries.
- **Untrusted model files**: never load a pickle from an untrusted source; loading can
  execute code.

Privacy:

- Send only the fields needed for prediction.
- Avoid logging raw personal data.
- Aggregate or hash identifiers.
- Define retention and deletion policies.
- Consider membership inference and re-identification risks.
- Comply with applicable regulations and organizational policy.

## 11. Deployment Patterns

- **Container**: package the app and dependencies into an image; run it anywhere.
- **Serverless**: run on demand; good for spiky, low-volume traffic; cold starts hurt
  latency.
- **Managed endpoint**: a cloud service loads the model and handles scaling.
- **Edge**: run on the user's device; low latency and privacy, limited compute.
- **Streaming**: consume events and emit predictions continuously.

The right pattern depends on latency, volume, cost, privacy, and operational capacity.
Start simple. A single containerized service with a health check is a valid production
starting point for many applications.

## 12. Common Mistakes

- **Fitting preprocessing in the serving code instead of loading the pipeline.**
  Training-serving skew.
- **Loading the model on every request.** Unnecessary latency.
- **No input validation.** Bad data produces silent nonsense.
- **Returning 200 for errors.** Clients cannot react correctly.
- **Hard-coded model paths or credentials.** Fragile and insecure.
- **No health check.** Orchestrators cannot tell whether the service is ready.
- **No version in the response.** Debugging becomes guesswork.
- **No monitoring.** Drift is discovered by users.
- **Logging personal data.** Privacy and legal risk.
- **No rollback plan.** A bad model stays live.
- **Ignoring latency.** A good model that is too slow is not deployable.

## 13. Summary

- Batch prediction computes many predictions on a schedule; real-time serves on
  demand.
- An API is a contract: input schema, output schema, status codes, version.
- Validate input at the boundary.
- Save and load the entire pipeline; pin versions.
- Training-serving consistency is the most common deployment failure.
- Load the model once, monitor latency and throughput.
- Drift and concept change require monitoring and retraining.
- Security, privacy, and rate limiting are part of deployment.
- Keep a rollback path and version every model.

## Outcome

Learners can explain batch and real-time serving, design a prediction API contract,
serialize and load complete pipelines, prevent training-serving skew, reason about
latency and throughput, plan monitoring and retraining, and apply security and privacy
practices to a deployed model.
