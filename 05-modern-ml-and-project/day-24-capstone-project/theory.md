# Day 24 — Theory: Capstone Review and Assessment

**Module:** Module 6: Modern ML and Project

The capstone is where the course stops being a list of techniques and becomes a
project. You choose a problem, build a pipeline, evaluate it honestly, and defend the
result to people who did not write the code. The technical work matters, but so does
the explanation: a model nobody trusts or understands will not be used.

This day reviews the complete workflow, gives criteria for choosing algorithms and
metrics, catalogs the mistakes that most often invalidate a result, and describes how
to communicate and assess a project.

## In Plain English

**Think of it like this:** the capstone is the whole course in one project — a
real question, clean data, a tuned model, an honest score, and a clear story.

- Define the **problem and success metric** before modeling.
- Explore and clean the data, then build a **simple baseline** first.
- Try a few models, tune with cross-validation, and compare them fairly.
- Evaluate **once** on a held-out test set and report uncertainty.
- Explain the results and limitations to a non-technical audience.

## 1. What the Capstone Demonstrates

A strong capstone shows that you can:

- Define a problem precisely, including the decision it supports.
- Find, inspect, and split data without leakage.
- Build a reproducible preprocessing and modeling pipeline.
- Establish a baseline and improve on it deliberately.
- Choose a metric that matches the cost of errors.
- Evaluate on data you did not use for training or tuning.
- Explain what the model learned and where it fails.
- Communicate the result to both technical and nontechnical audiences.
- Identify limitations, risks, and appropriate use.

The project does not need a novel algorithm. It needs a defensible process.

## 2. The Complete Workflow

```text
1. Problem definition
2. Data collection and inspection
3. Split: train / validation / test
4. Preprocessing pipeline
5. Baseline
6. Candidate models and tuning
7. Validation-based selection
8. Single test evaluation
9. Error analysis and subgroup checks
10. Serialization and documentation
11. Presentation and defense
```

At each step, ask what could go wrong and how you would detect it. The assessment is
as much about the reasoning as the final score.

### Problem definition

- What is the unit of prediction?
- What is the target?
- What information is available at prediction time?
- What decision does the prediction support?
- What happens when the model is wrong?
- What are the latency, cost, privacy, and fairness constraints?

### Data inspection

- Shape, types, missing values, duplicates.
- Target distribution, class imbalance.
- Leakage-prone columns, such as IDs or post-outcome fields.
- Group and time structure.
- Obvious data quality problems.

### Splitting

- Stratify classification targets.
- Group by entity when rows repeat.
- Split time-ordered data chronologically.
- Hold out the test set before any performance-driven decisions.

### Preprocessing

- Impute, encode, and scale inside a pipeline.
- Use `handle_unknown="ignore"` for production encoders.
- Fit every learned transformation inside each fold.
- Document decisions and their rationale.

### Baseline

- Regression: predict the mean or median.
- Classification: predict the majority class.
- Report the baseline alongside every model.
- A model that barely beats the baseline demands an explanation.

### Model selection

- Compare a small set of sensible candidates.
- Use the same splits, metric, and seeds.
- Prefer simpler models unless complexity earns a clear gain.
- Record the comparison, not just the winner.

### Final evaluation

- Evaluate once on the test set.
- Report the primary metric and supporting metrics.
- Include uncertainty or fold-to-fold variation.
- State the threshold and `pos_label`.
- Show the confusion matrix or residuals.

## 3. Choosing an Algorithm

There is no universal best model. Match the algorithm to the data and constraints.

| Situation | Reasonable starting point |
|-----------|---------------------------|
| Small/medium tabular data | logistic regression, random forest, gradient boosting |
| High-dimensional sparse data | linear models with regularization |
| Nonlinear tabular patterns | gradient boosting, random forest |
| Interpretability required | linear model, small tree, monotonic model |
| Images | CNN or a pretrained vision model |
| Text | TF-IDF + linear model, or a pretrained transformer |
| Time series | seasonal baselines, gradient boosting with lag features, specialized models |
| Very large data | scalable linear models, SGD, distributed methods |

Practical rules:

- Start simple, establish a baseline, then add complexity.
- Prefer models whose assumptions fit the data.
- Prefer models you can evaluate and explain within the project's constraints.
- If two models perform similarly, choose the simpler and cheaper one.

## 4. Evaluation Checklist

### Metric fit

- Does the metric match the cost of errors?
- Is the metric appropriate for class balance?
- Is the averaging (macro, weighted, micro) stated?
- Is the threshold justified, or is the full curve reported?

### Estimation quality

- Was the test set untouched during development?
- Were preprocessing and tuning inside the CV loop?
- Were groups and time respected?
- Is the score stable across folds or seeds?
- Is the baseline reported?

### Error analysis

- Which examples are misclassified, and why?
- Are errors concentrated in a subgroup or region?
- Is the model failing on a class that matters?
- Are residuals biased or heteroscedastic?

### Assumptions

- Is the test distribution representative of deployment?
- Are there features that would not be available in production?
- Is the target measured consistently over time?

## 5. Common Mistakes

### Leakage

- Preprocessing or feature selection fitted on all data.
- Target encoding computed outside the folds.
- Future information in features.
- Duplicate records across train and test.
- Tuning on the test set.

### Overfitting

- Too many features for too few rows.
- Hyperparameters tuned to noise.
- A test score far above the CV score.
- Feature importance dominated by a single unstable feature.

### Inappropriate metrics

- Accuracy on imbalanced data.
- AUC presented without precision.
- RMSE compared across datasets with different scales.
- A single number with no context or baseline.

### Unreliable test results

- A test set used many times during development.
- A very small test set read as precise.
- No stratification or group awareness.
- Reporting the best fold instead of the mean.

### Communication failures

- No problem statement.
- No baseline.
- Hiding limitations.
- Claiming causation from correlation.
- Presenting a polished demo of only the cases that work.

## 6. Limitations and Uncertainty

Every model has limits. State them:

- **Scope**: the population and conditions where the model was validated.
- **Data**: what was not represented, and how that could bias predictions.
- **Performance**: expected metric range, not a single cherry-picked score.
- **Failure modes**: inputs or subgroups where the model is unreliable.
- **Drift**: conditions that would require retraining.
- **Ethics**: who could be harmed, and what safeguards exist.

Be explicit about uncertainty. "The model achieves 0.92 recall with a 95% range of
0.88 to 0.95 across folds" is more honest than "the model is 92% accurate."

## 7. Communicating to Different Audiences

### Technical audience

- Show the pipeline, splits, and evaluation protocol.
- Discuss leakage checks and why the metric was chosen.
- Report CV and test results with variation.
- Show the confusion matrix, residuals, or calibration.
- Explain model choice and rejected alternatives.

### Nontechnical audience

- Lead with the problem and the decision.
- Use a concrete example of a correct and an incorrect prediction.
- Explain the metric in terms of consequences, not formulas.
- State what the model cannot do.
- Avoid jargon; define it when unavoidable.
- Show a demo on realistic inputs.

Both audiences need the same honesty about limitations. A nontechnical audience is
not served by hiding uncertainty; it is served by translating it.

## 8. Responsible Use

Ask before deployment:

- **Fairness**: does performance differ across groups, and is the difference
  acceptable?
- **Privacy**: is personal data minimized, secured, and retained appropriately?
- **Transparency**: can affected people understand and challenge a decision?
- **Accountability**: who owns the model, and who responds when it fails?
- **Appropriate use**: is the model being applied within its validated scope?
- **Human oversight**: is there a path for review in high-stakes decisions?

A capstone should include at least a short section on these questions, even if the
project is not deployed.

## 9. What a Strong Submission Looks Like

- A clear problem statement and success metric.
- A reproducible repository or notebook with fixed seeds.
- A documented data source and split strategy.
- A pipeline that includes preprocessing.
- A baseline and at least two candidate models.
- Honest CV and single test evaluation.
- Error analysis, including subgroups.
- A saved model and usage instructions.
- A model card or README with limitations and intended use.
- A presentation that is understandable to a mixed audience.

A weaker submission often has a high score and a thin process. The assessment rewards
the process because the process is what transfers to new problems.

## 10. Defending the Work

Expect questions such as:

- Why this metric, and what does it miss?
- How do you know the test set was not contaminated?
- What happens to performance on a different population?
- Which errors are most costly, and how does the model handle them?
- What would you do with more data or more time?
- Why did you reject the other models?
- What is the model's failure mode?

Good answers are specific and honest. "We have not tested that" is acceptable when
followed by a plan. Overclaiming is the fastest way to lose credibility.

## 11. Summary

- The capstone demonstrates a defensible process, not a novel algorithm.
- Define the problem, split before evaluating, and pipeline the preprocessing.
- Establish a baseline and choose a metric that matches error costs.
- Select on validation, evaluate once on test, and report uncertainty.
- Analyze errors and subgroups.
- Document limitations, intended use, and ethical considerations.
- Communicate differently to technical and nontechnical audiences.
- Defend the reasoning, not just the score.

## Outcome

Learners can present a complete ML project, justify algorithm and metric choices,
demonstrate a leak-free evaluation, analyze errors and subgroups, communicate results
to mixed audiences, state limitations and ethical considerations, and answer
conceptual and practical questions about their work.
