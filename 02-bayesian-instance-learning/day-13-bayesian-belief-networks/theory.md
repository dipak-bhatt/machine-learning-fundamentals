# Day 13 — Theory: Bayesian Belief Networks

**Module:** Module 3: Bayesian and Instance-Based Learning

Naive Bayes assumes every feature is independent given the class. That assumption
keeps the model simple, but it ignores real structure: a disease causes several
symptoms, a cloudy sky makes both rain and a full sprinkler more likely, and a
customer's income influences both their spending and their churn. A **Bayesian
belief network** (also called a Bayesian network) represents those dependencies
directly as a graph.

The graph is a **directed acyclic graph (DAG)**. Nodes are random variables; arrows
mean "directly influences". Each node carries a **conditional probability table**
that says how it behaves given its parents. From these local tables the network
encodes a full joint distribution over all variables, and from that we can answer
probabilistic questions such as "given that the grass is wet, how likely is rain?"
or "given the alarm, did a burglary occur?"

## In Plain English

**Think of it like this:** a Bayesian network is a map of which things influence
which, with a small probability table attached to each part of the map.

- **Nodes** are variables; **arrows** mean "directly influences"; no cycles are
  allowed.
- Each node stores a **conditional probability table** given its parents.
- The graph encodes independence assumptions and makes the full joint
  distribution compact.
- Given evidence, we can ask "how likely is X?" using inference.
- The arrows need domain knowledge; a wrong map gives wrong answers.

## 1. From Naive Bayes to a Network

Naive Bayes has a star-shaped graph:

```text
        Class
       /  |  \
   X1   X2   X3
```

All features depend on the class, and no feature depends on another. A Bayesian
network generalizes this: any variable can depend on any set of other variables, as
long as the arrows do not form a cycle.

```text
   Rain        Sprinkler
      \          /
       \        /
        WetGrass
```

Now rain and sprinkler are allowed to interact. The graph says both influence
whether the grass is wet, and it says nothing about a relationship between rain and
sprinkler, so they are treated as independent causes here.

The graph is not just a drawing. It encodes assumptions: which variables are
conditionally independent, which evidence can affect which query, and how the joint
distribution factorizes.

## 2. Nodes, Edges, and the DAG

The components are:

- **Nodes** represent random variables, either discrete categories or continuous
  values. In this course we use discrete variables with a small number of states.
- **Directed edges** go from a **parent** to a **child**. An arrow from `A` to `B`
  means `A` is a direct cause or predictor of `B`.
- **Acyclic** means there is no directed path that returns to its starting node.
  Cycles would create circular definitions, so they are forbidden.
- **Root nodes** have no parents and carry a prior distribution.
- **Leaf nodes** have no children.

The **parents** of a node are the nodes with an arrow pointing into it. The
**descendants** are nodes reachable by following arrows forward.

In the example, `Rain` and `Sprinkler` are roots. `WetGrass` has parents `Rain` and
`Sprinkler`. A common larger example adds `Cloudy` as a parent of both `Rain` and
`Sprinkler`.

## 3. Conditional Probability Tables

Each node stores one probability for every combination of its parents' states. This
is a **conditional probability table (CPT)**.

For a root node, the CPT is just a prior. For `Rain`:

| Rain | P |
|------|---|
| true | 0.2 |
| false | 0.8 |

For `WetGrass`, the CPT has one row per combination of `Rain` and `Sprinkler`:

| Rain | Sprinkler | P(WetGrass = true) |
|------|-----------|--------------------|
| true | true | 0.99 |
| true | false | 0.90 |
| false | true | 0.90 |
| false | false | 0.01 |

The probabilities of `WetGrass = false` are the complements. A CPT must have
probabilities that sum to 1 across the states of the child, for every parent
combination.

The size of a CPT grows exponentially with the number of parents: with `k` parents
of binary states, a binary child needs `2^k` numbers. This is why networks use
sparse, local dependencies rather than a single giant table.

## 4. The Joint Distribution Factorizes

The central result is that the joint probability of an assignment factors into one
term per node, each conditioned only on its parents:

```text
P(X_1, ..., X_n) = product_i P(X_i | parents(X_i))
```

For the weather network:

```text
P(Rain, Sprinkler, WetGrass)
  = P(Rain) * P(Sprinkler) * P(WetGrass | Rain, Sprinkler)
```

This is a massive compression. A full joint table over three binary variables needs
`2^3 - 1 = 7` independent numbers. The factorized form needs:

- `Rain`: 1 number,
- `Sprinkler`: 1 number,
- `WetGrass`: 4 numbers,

for 6 numbers total, and the saving grows quickly with more variables. Each CPT can
be estimated from data or elicited from an expert, one small table at a time.

### Computing one joint entry

For `Rain = true`, `Sprinkler = true`, `WetGrass = true`:

```text
P = 0.2 * 0.5 * 0.99 = 0.099
```

For `Rain = false`, `Sprinkler = false`, `WetGrass = false`:

```text
P = 0.8 * 0.5 * 0.99 = 0.396
```

The product is not a probability of a single event in isolation; it is the
probability of that complete assignment.

## 5. Conditional Independence

The graph encodes conditional independence through a rule called **d-separation**.
The intuition is enough for this course:

- Two variables are generally dependent if a path of arrows connects them.
- A path is **blocked** if it passes through a node whose value is known (a
  **collider** is the exception).
- If all paths between two variables are blocked, they are conditionally
  independent given the observed nodes.

Three important patterns:

**Chain:** `A -> B -> C`. If `B` is known, `A` and `C` are independent. Knowing the
middle explains the ends.

**Fork:** `A <- B -> C`. If `B` is known, `A` and `C` are independent. A common
cause explains the correlation.

**Collider:** `A -> B <- C`. Here `A` and `C` are independent unless `B` (or a
descendant of `B`) is observed. Observing the common effect creates a dependence
between the causes. This is the **explaining away** phenomenon.

In the weather network, `Rain` and `Sprinkler` are independent. Once we observe
`WetGrass`, they become dependent: a wet lawn can be explained by rain or by the
sprinkler, so evidence for one reduces the probability of the other.

## 6. Inference by Enumeration

**Inference** means computing a probability of some variables given evidence. For a
small network, the simplest method is **enumeration**: list every complete
assignment, keep the ones consistent with the evidence, and sum their joint
probabilities.

For a query `P(Q | E = e)`:

```text
P(Q | e) = P(Q, e) / P(e)
```

where both numerator and denominator are sums of joint probabilities over the
assignments that match.

Example: `P(Rain | WetGrass = true)`.

1. Compute `P(WetGrass = true)` by summing the joint over all `Rain` and
   `Sprinkler` values with `WetGrass = true`.
2. Compute `P(Rain = true, WetGrass = true)` similarly, restricting `Rain = true`.
3. Divide.

Enumeration is exact and easy to implement, but its cost grows exponentially with
the number of variables. Real networks use variable elimination, belief
propagation, or sampling. The principle is the same.

## 7. Explaining Away

The hallmark of a Bayesian network is the ability to reason in both directions and
to handle **explaining away**.

Suppose the grass is wet and the sprinkler is on. Rain and sprinkler both explain
the wetness. If we already know the sprinkler is on, rain becomes *less* likely than
it would be with no explanation, because the wetness is already accounted for.

In numbers, with the CPT above:

```text
P(Rain | WetGrass)              = 0.342
P(Rain | WetGrass, Sprinkler)   = 0.216
```

Observing the sprinkler lowered the probability of rain from 34% to 22%. That is
not something a simple Naive Bayes model does; it requires representing the shared
effect.

Conversely, if the sprinkler is off and the grass is wet, rain becomes much more
likely:

```text
P(Rain | WetGrass, Sprinkler = false) = 0.957
```

The observation is "explained" by the only remaining cause.

## 8. Bayesian Networks versus Naive Bayes

| Aspect | Naive Bayes | Bayesian network |
|--------|-------------|------------------|
| Graph | class -> every feature | arbitrary DAG |
| Dependencies | features independent given class | modelled explicitly |
| Parameters | one table per feature per class | one CPT per node |
| Inference | direct scoring | enumeration or message passing |
| Interpretability | word or feature weights | graph plus CPTs |
| Flexibility | fixed structure | can encode domain knowledge |

Naive Bayes is a special case of a Bayesian network with a single class parent and
no feature-feature edges. Moving to a general network buys realism at the cost of
more parameters and more complex inference.

## 9. Where the Numbers Come From

CPTs can be obtained in three ways:

- **Expert elicitation.** A clinician or engineer states the probabilities. Useful
  when data are scarce.
- **Learning from data.** With a known structure, each CPT is estimated by counting
  parent-child combinations, with smoothing for rare ones.
- **Structure learning.** When the graph is unknown, algorithms search over possible
  DAGs using a score such as Bayesian information criterion (BIC) or a
  conditional-independence test. This is harder and needs more data.

In practice, domain experts often propose the graph and data fill in the tables.
The graph is a modelling decision, not a fact discovered entirely from data.

## 10. Applications

Bayesian networks shine when variables have clear causal or diagnostic
relationships and uncertainty must be handled explicitly:

- **Medical diagnosis:** symptoms, risk factors, and diseases with CPTs estimated
  from studies. The graph supports reasoning from symptoms to causes and from causes
  to likely symptoms.
- **Fault diagnosis:** equipment components and observed alarms. Explaining away
  helps separate a failed sensor from a failed subsystem.
- **Risk analysis:** dependencies among failure events, used in safety and finance.
- **Decision support:** combined with utilities to choose actions under uncertainty.
- **Bioinformatics and genetics:** regulatory relationships among genes.

They are less suited to problems with thousands of densely connected variables,
where learning and inference become expensive.

## 11. Strengths and Limitations

**Strengths**

- Represents dependencies and causal structure explicitly.
- Handles missing data naturally by marginalizing.
- Supports both predictive and diagnostic reasoning.
- Combines expert knowledge with data.
- Gives interpretable probabilities and explanations.
- Can answer many queries with one model.

**Limitations**

- Exact inference is exponential in the worst case.
- Structure learning is difficult and data-hungry.
- CPTs grow with the number of parents.
- Requires careful specification of states and priors.
- Continuous variables need discretization or special distributions.
- A wrong graph produces confidently wrong answers.

## 12. Common Mistakes

- **Creating a cycle.** The graph must be a DAG; cycles make the factorization
  invalid.
- **Confusing correlation with causation.** An arrow encodes a modelling assumption,
  not proof of causation.
- **Forgetting to normalize.** The numerator of Bayes' rule must be divided by the
  evidence.
- **Ignoring explaining away.** Adding a cause as evidence changes the probability
  of the others.
- **Over-dense graphs.** Every extra parent multiplies the CPT size.
- **Using exact enumeration on a large network.** Switch to a proper inference
  algorithm.
- **Treating learned structure as truth.** Structure learning is sensitive to
  sample size and scoring choices.

## 13. Summary

- A Bayesian network is a DAG whose nodes are variables and whose edges are direct
  dependencies.
- Each node has a CPT conditioned on its parents.
- The joint distribution factorizes into local CPTs, which is a large compression.
- The graph encodes conditional independence and enables explaining away.
- Inference by enumeration sums joint probabilities consistent with the evidence.
- Naive Bayes is a star-shaped special case.
- Networks are powerful for diagnosis and risk, but inference and learning can be
  expensive.

## Outcome

Learners can read and construct a small DAG, write conditional probability tables,
factorize a joint distribution from the graph, explain conditional independence and
explaining away, perform exact inference by enumeration, and compare Bayesian
networks with Naive Bayes.
