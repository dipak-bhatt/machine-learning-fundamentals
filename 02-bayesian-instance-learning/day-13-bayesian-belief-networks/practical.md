# Day 13 — Practical: Bayesian Belief Networks

**Module:** Module 3: Bayesian and Instance-Based Learning

In this practical we build a small Bayesian network from scratch. We will:

- Specify the DAG and its conditional probability tables.
- Enumerate the full joint distribution and check that it sums to 1.
- Perform exact inference by enumeration.
- Compute diagnostic probabilities such as `P(rain | wet grass)`.
- Observe explaining away when the sprinkler is added as evidence.

Every calculation uses plain Python and NumPy, so no external probabilistic
programming library is required.

## 1. Setup

```python
import itertools

import numpy as np
import pandas as pd

RANDOM_STATE = 42
```

The variable names below are `rain`, `sprinkler`, and `wet`. Each is Boolean, which
keeps the tables small enough to print in full.

## 2. Define the Network

The graph is:

```text
   Rain        Sprinkler
      \          /
       \        /
        WetGrass
```

`Rain` and `Sprinkler` are independent causes of `WetGrass`. The CPTs are:

```python
P_RAIN = {True: 0.20, False: 0.80}
P_SPRINKLER = {True: 0.50, False: 0.50}

P_WET = {
    (True, True): {True: 0.99, False: 0.01},
    (True, False): {True: 0.90, False: 0.10},
    (False, True): {True: 0.90, False: 0.10},
    (False, False): {True: 0.01, False: 0.99},
}
```

The keys of `P_WET` are `(rain, sprinkler)` pairs. Reading one row: if it is raining
and the sprinkler is on, the grass is wet with probability 0.99. If neither cause is
active, the grass is still wet with probability 0.01, representing dew or a
mislabelled observation.

The full joint distribution factorizes as:

```text
P(Rain, Sprinkler, WetGrass)
  = P(Rain) * P(Sprinkler) * P(WetGrass | Rain, Sprinkler)
```

```python
def joint_probability(rain, sprinkler, wet):
    return (
        P_RAIN[rain]
        * P_SPRINKLER[sprinkler]
        * P_WET[(rain, sprinkler)][wet]
    )
```

## 3. Enumerate the Joint Distribution

There are only `2^3 = 8` complete assignments. Enumerate them and check the
probabilities sum to 1.

```python
rows = []

for rain, sprinkler, wet in itertools.product([True, False], repeat=3):
    rows.append(
        {
            "rain": rain,
            "sprinkler": sprinkler,
            "wet": wet,
            "probability": joint_probability(rain, sprinkler, wet),
        }
    )

joint_table = pd.DataFrame(rows)
print(joint_table.to_string(index=False))
print("total probability:", round(joint_table["probability"].sum(), 6))
```

Expected output:

```text
 rain  sprinkler   wet  probability
 True       True  True        0.099
 True       True False        0.001
 True      False  True        0.090
 True      False False        0.010
False       True  True        0.360
False       True False        0.040
False      False  True        0.004
False      False False        0.396
total probability: 1.0
```

The table is the entire model in tabular form. Every inference question is a sum
over the rows consistent with the evidence.

## 4. Inference by Enumeration

Write one function that computes `P(variable = value | evidence)`.

```python
def infer(variable, value, evidence):
    total = 0.0
    matching = 0.0

    for rain, sprinkler, wet in itertools.product([True, False], repeat=3):
        assignment = {"rain": rain, "sprinkler": sprinkler, "wet": wet}

        if any(assignment[key] != observed for key, observed in evidence.items()):
            continue

        probability = joint_probability(rain, sprinkler, wet)
        total += probability

        if assignment[variable] == value:
            matching += probability

    return matching / total if total > 0 else 0.0
```

The function does three things:

1. Skips assignments that contradict the evidence.
2. Sums the probability of the remaining assignments to get `P(evidence)`.
3. Sums the subset where the query is true to get `P(query, evidence)`.

The ratio is the conditional probability.

## 5. Marginal and Conditional Probabilities

Start with the marginals.

```python
print("P(rain):", round(infer("rain", True, {}), 4))
print("P(sprinkler):", round(infer("sprinkler", True, {}), 4))
print("P(wet):", round(infer("wet", True, {}), 4))
```

Expected output:

```text
P(rain): 0.2
P(sprinkler): 0.5
P(wet): 0.553
```

The prior for rain is 0.2, and the prior for the sprinkler is 0.5. The marginal
probability of wet grass is 0.553, which is higher than either cause alone because
either one can wet the grass.

Now condition on the grass being wet:

```python
print("P(rain | wet):", round(infer("rain", True, {"wet": True}), 4))
print("P(sprinkler | wet):", round(infer("sprinkler", True, {"wet": True}), 4))
```

Expected output:

```text
P(rain | wet): 0.3418
P(sprinkler | wet): 0.83
```

Rain was 20% a priori and rises to 34% after seeing wet grass. The sprinkler was
50% and rises to 83%. Both causes become more likely because both explain the
observation. Inference runs in the diagnostic direction, from effect to cause,
which is exactly what Bayes' theorem is for.

## 6. Explaining Away

Now add the sprinkler to the evidence. If the sprinkler already explains the wet
grass, rain should become *less* likely than it was with wet grass alone.

```python
print(
    "P(rain | wet, sprinkler):",
    round(infer("rain", True, {"wet": True, "sprinkler": True}), 4),
)
print(
    "P(rain | wet, sprinkler=false):",
    round(infer("rain", True, {"wet": True, "sprinkler": False}), 4),
)
```

Expected output:

```text
P(rain | wet, sprinkler): 0.2157
P(rain | wet, sprinkler=false): 0.9574
```

Compare the three numbers:

| Evidence | `P(rain)` |
|----------|-----------|
| none | 0.200 |
| wet grass | 0.342 |
| wet grass and sprinkler on | 0.216 |
| wet grass and sprinkler off | 0.957 |

Observing the sprinkler *reduces* the probability of rain from 0.342 to 0.216. The
sprinkler has taken over as the explanation. This is **explaining away**, and it is
the signature of a collider: `Rain -> WetGrass <- Sprinkler`. Rain and sprinkler
are independent a priori, but they become dependent once their shared effect is
observed.

When the sprinkler is known to be off, the only remaining explanation is rain, so
the probability jumps to 96%.

One more check: if the grass is dry, rain should become unlikely.

```python
print("P(rain | not wet):", round(infer("rain", True, {"wet": False}), 4))
```

Expected output:

```text
P(rain | not wet): 0.0246
```

Rain falls from 20% to 2.5%. The network reasons correctly in both directions.

## 7. Reading the Evidence Path

A Bayesian network explanation is a set of probabilities, not a single path. But we
can still describe the reasoning:

- The prior says rain is uncommon (20%) and the sprinkler is a coin flip (50%).
- Wet grass is common overall (55.3%) because it has two causes.
- Seeing wet grass raises both causes.
- Seeing the sprinkler on lowers the need for rain, because one explanation is
  enough.

This is closer to how a person reasons than a single coefficient. It also shows why
observing a common effect can change beliefs about independent causes.

## 8. Optional: The Same Network with `pgmpy`

If `pgmpy` is installed, the same model can be expressed with a library:

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork(
    [("rain", "wet"), ("sprinkler", "wet")]
)

cpd_rain = TabularCPD(
    variable="rain", variable_card=2, values=[[0.8], [0.2]]
)
cpd_sprinkler = TabularCPD(
    variable="sprinkler", variable_card=2, values=[[0.5], [0.5]]
)
cpd_wet = TabularCPD(
    variable="wet",
    variable_card=2,
    values=[
        [0.99, 0.10, 0.10, 0.01],
        [0.01, 0.90, 0.90, 0.99],
    ],
    evidence=["rain", "sprinkler"],
    evidence_card=[2, 2],
)

model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet)
print("model valid:", model.check_model())

inference = VariableElimination(model)
result = inference.query(
    variables=["rain"], evidence={"wet": 1}, show_progress=False
)
print(result)
```

The library produces the same 0.342 answer. The manual implementation is worth
doing once because it makes clear that inference is just sums of joint
probabilities; the library is faster and scales to larger graphs.

## 9. Try It Yourself

1. **Add Cloudy.** Introduce `Cloudy` with prior 0.3, make it a parent of both
   `Rain` and `Sprinkler`, and update the CPTs. Compute `P(rain | wet)` again.
2. **Change the prior.** Set `P(rain=True)=0.05`. How does `P(rain | wet)` change?
3. **Add a second effect.** Add `Slippery` as a child of `Rain` only. Does observing
   `Slippery` change `P(sprinkler | wet)`? Explain why.
4. **Remove a cause.** Delete the `sprinkler -> wet` edge and recompute. What
   happens to explaining away?
5. **Make a cycle.** Try to add `wet -> rain` and explain why the factorization
   would no longer be valid.
6. **Scale it up.** Estimate how many rows a network with 10 binary variables would
   need for full enumeration, and why real systems avoid it.

## 10. What We Learned

- A Bayesian network is a DAG plus a CPT for each node.
- The joint distribution factorizes into local conditional probabilities.
- Enumerating the joint table gives exact answers for a small network.
- Inference runs from effects to causes as well as from causes to effects.
- Observing a common effect creates dependence between independent causes.
- Explaining away makes a cause less likely once another cause is confirmed.
- Exact enumeration is simple but exponential; real networks use smarter
  algorithms.

## Outcome

Learners can specify a small Bayesian network, compute its joint distribution,
perform exact inference by enumeration, interpret diagnostic and predictive
probabilities, and explain conditional independence and explaining away in a
concrete example.
