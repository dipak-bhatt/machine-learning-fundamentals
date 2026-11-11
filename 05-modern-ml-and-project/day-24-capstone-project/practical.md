# Day 24 — Practical: Capstone Demonstration and Assessment

**Module:** Module 6: Modern ML and Project

This practical is the capstone itself. You will choose a problem, build and document a
complete pipeline, demonstrate it on unseen records, present it, and answer questions.
The goal is a defensible project, not a perfect score.

The day includes:

- A project brief and suggested options.
- A verified reference skeleton you can adapt.
- Reproducibility and code-review checklists.
- A model-card template.
- A presentation outline and demo script.
- Peer-review and viva question banks.
- An assessment rubric.

## 1. Choose a Project

Pick a problem that is small enough to finish and rich enough to show the workflow.
Good options:

1. **House price prediction.** Use `house_prices.csv`. Regression, missing values,
   categorical encoding, RMSE.
2. **Breast cancer classification.** Use `load_breast_cancer`. Binary classification,
   recall for the malignant class, threshold trade-offs.
3. **Digit recognition.** Use `load_digits`. Multi-class classification, confusion
   matrix, error inspection.
4. **Customer churn (synthetic).** Generate mixed-type data with
   `make_classification`, add categorical columns and missingness. Binary
   classification, class imbalance, subgroup audit.
5. **Iris species classification.** Small and clean; best for focusing on process and
   communication rather than model complexity.
6. **A dataset of your own.** Allowed if it is small, legally usable, and has a clear
   target.

Avoid projects where the target leaks the answer, where rows are duplicated across
time, or where the data cannot be shared with the reviewer.

## 2. Required Deliverables

Submit:

- **Code**: a notebook or scripts that run end to end.
- **README**: problem, data, method, results, how to run.
- **Model artifact**: a saved pipeline (`joblib`) and the code that produced it.
- **Model card**: intended use, training data, metrics, limitations, ethical notes.
- **Presentation**: 10–12 minutes plus questions.
- **Demo**: predictions on at least three unseen records.

The README and model card are part of the assessment. A good model with no explanation
is an incomplete project.

## 3. Reference Skeleton

The following skeleton is complete and verified on breast cancer. Adapt the data
loading, target, and metrics to your project.

```python
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

print("CV folds:", np.round(scores, 4))
print("CV mean:", round(scores.mean(), 4))
print("CV std:", round(scores.std(), 4))
```

Expected output:

```text
CV folds: [0.967 0.989 0.978 0.989 0.967]
CV mean: 0.978
CV std: 0.0098
```

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("test accuracy:", round(accuracy_score(y_test, predictions), 4))
print(confusion_matrix(y_test, predictions))
print(classification_report(
    y_test, predictions, target_names=cancer.target_names, digits=4
))
```

Expected output:

```text
test accuracy: 0.9825
[[41  1]
 [ 1 71]]
              precision    recall  f1-score   support

   malignant     0.9762    0.9762    0.9762        42
      benign     0.9861    0.9861    0.9861        72

    accuracy                         0.9825       114
   macro avg     0.9812    0.9812    0.9812       114
weighted avg     0.9825    0.9825    0.9825       114
```

Save the pipeline and verify it reloads:

```python
joblib.dump(model, "capstone_model.joblib")
loaded = joblib.load("capstone_model.joblib")

same = np.allclose(loaded.predict(X_test), predictions)
print("reload matches:", same)
```

Expected output:

```text
reload matches: True
```

Remember the `pos_label` issue from Day 18. If malignant is the class you most want
to catch, compute recall with `pos_label=0` and report it explicitly.

## 4. Reproducibility Checklist

- [ ] One command or notebook runs the full pipeline.
- [ ] `RANDOM_STATE` is defined once and passed everywhere.
- [ ] Library versions are recorded.
- [ ] The data source and version are documented.
- [ ] The split is reproducible.
- [ ] Preprocessing is inside the pipeline.
- [ ] The test set is not used during development.
- [ ] The saved model is the same object that produced the reported scores.
- [ ] The README states how to reproduce the results.

If a reviewer cannot rerun your project and get the same numbers, the result is not yet
defensible.

## 5. Code Review Checklist

### Data

- [ ] Target and features are clearly separated.
- [ ] Missing values are handled explicitly.
- [ ] Duplicates and leakage-prone columns are checked.
- [ ] Class or target distribution is reported.
- [ ] Split strategy respects groups and time.

### Pipeline

- [ ] Imputation, encoding, and scaling are inside the pipeline.
- [ ] `handle_unknown="ignore"` is used for production encoders.
- [ ] No transformation is fitted on the full dataset.
- [ ] The pipeline is saved as a whole.

### Modeling

- [ ] A baseline is reported.
- [ ] At least two candidates are compared.
- [ ] The same CV scheme is used for all candidates.
- [ ] Hyperparameters are tuned on training data only.
- [ ] The final model is chosen on validation, not test.

### Evaluation

- [ ] The metric matches the decision.
- [ ] The threshold and `pos_label` are stated.
- [ ] CV mean and spread are reported.
- [ ] The test set is evaluated once.
- [ ] Confusion matrix or residuals are shown.
- [ ] Errors and subgroups are analyzed.

## 6. Error Analysis Template

Fill this in for your project:

```text
Total errors: ___
Errors by class / target range: ___
Most confident errors: ___
Errors by subgroup (region, plan, sex, age band): ___
Common pattern in errors: ___
Likely cause (data, features, label noise, model capacity): ___
One change that might fix it: ___
```

Inspect at least five individual errors. If possible, show one in the presentation.

## 7. Model Card Template

```text
Model name:
Version:
Owner:
Date:
Intended use:
Out-of-scope uses:
Training data:
Features:
Target:
Split strategy:
Model type:
Primary metric and value:
Secondary metrics:
Threshold / pos_label:
Performance by subgroup:
Known limitations:
Ethical considerations:
How to reproduce:
```

The model card is short but forces the questions a reviewer will ask.

## 8. Presentation Outline

Aim for 10–12 minutes:

1. **Problem** (1 min): decision, unit, target, why it matters.
2. **Data** (2 min): source, size, target distribution, missingness, split.
3. **Method** (2 min): preprocessing, baseline, candidates, selection.
4. **Evaluation** (3 min): CV and test results, confusion matrix or residuals,
   uncertainty.
5. **Error analysis** (2 min): who or what the model gets wrong.
6. **Limitations and responsible use** (1 min): scope, fairness, privacy, monitoring.
7. **Demo** (1–2 min): live predictions on unseen records.
8. **Lessons** (1 min): what you would do next.

Slides should be readable, not dense. Put detailed tables in an appendix.

## 9. Live Demo Script

Prepare three unseen records before the presentation:

```python
new_records = X_test[:3]
predictions = loaded.predict(new_records)
print(predictions)
```

For each record, state:

- The input values that matter.
- The prediction.
- Whether it is correct, if the label is known.
- Why the prediction is plausible or surprising.

A demo that only shows correct cases looks staged. Show at least one error or
uncertain case and explain it.

## 10. Peer Review Form

Use this to review another project:

```text
Project:
Reviewer:

1. Was the problem and metric clear? (1-5)
2. Was the data split free of obvious leakage? (1-5)
3. Was a baseline reported? (1-5)
4. Was the evaluation honest and complete? (1-5)
5. Were errors and limitations discussed? (1-5)
6. Could you reproduce the result from the README? (1-5)

Strongest part:
Most important improvement:
One question for the presenter:
```

Give specific, actionable feedback. "Good job" is not a review.

## 11. Viva Question Bank

Be ready for:

- Why did you choose this problem and metric?
- What is the unit of prediction, and when is the prediction made?
- How did you split the data, and why?
- How do you know there is no leakage?
- What does the baseline achieve?
- Why did you choose this model over the alternatives?
- What does the confusion matrix tell you?
- Which errors are most costly, and how does the model handle them?
- How stable are the results across folds or seeds?
- What happens on an input unlike the training data?
- Which subgroup performs worst, and what would you do about it?
- What are the privacy and fairness risks?
- What would you monitor after deployment?
- What would you do with more data, time, or compute?

Answer with specifics. It is fine to say "I did not test that; I would test it by…"

## 12. Assessment Rubric

| Criterion | Weight | What earns full credit |
|-----------|--------|------------------------|
| Problem framing | 10% | Clear decision, unit, target, metric, constraints |
| Data handling | 20% | Clean inspection, correct split, no leakage |
| Modeling | 20% | Baseline, sensible candidates, justified selection |
| Evaluation | 20% | Honest CV, single test, metric fit, error analysis |
| Reproducibility | 10% | Seeds, versions, runnable code, saved pipeline |
| Communication | 10% | Clear presentation, README, model card, demo |
| Responsible use | 10% | Limitations, fairness, privacy, monitoring plan |

A project with a high score but a contaminated evaluation should not pass the
evaluation criterion. Process failures outweigh headline numbers.

## 13. Final Checklist

- [ ] Problem statement and metric are written down.
- [ ] Data is inspected and documented.
- [ ] Split is reproducible and leak-free.
- [ ] Pipeline includes all preprocessing.
- [ ] Baseline is reported.
- [ ] Candidates are compared with the same protocol.
- [ ] Test set is evaluated once.
- [ ] Errors and subgroups are analyzed.
- [ ] Model and metadata are saved.
- [ ] README and model card are complete.
- [ ] Presentation fits the time limit.
- [ ] Demo uses unseen records.
- [ ] Limitations and responsible-use notes are included.

## 14. What We Learned

- A capstone is judged on process as much as performance.
- A clear problem statement and metric anchor every later decision.
- Honest splitting and pipelined preprocessing are non-negotiable.
- Baselines and simple models are part of a strong solution.
- Error analysis and subgroup checks reveal what averages hide.
- Reproducibility is a deliverable, not a footnote.
- Communication must be tailored to the audience.
- Responsible use requires explicit limitations and safeguards.
- The ability to defend the reasoning is the durable outcome.

## Outcome

Learners can plan and deliver a complete ML project, justify their data, modeling, and
evaluation choices, analyze errors and subgroups, document the model and its
limitations, present to mixed audiences, demonstrate predictions on unseen records, and
answer conceptual and practical questions about their work.
