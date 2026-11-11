# Day 13 — Bayesian Belief Networks

**Module:** Module 3: Bayesian and Instance-Based Learning

Naive Bayes assumes features are independent given the class. Real problems rarely
fit that star shape: diseases cause multiple symptoms, a cloudy sky raises both the
chance of rain and the chance of a full sprinkler, and one effect can have several
possible causes. A **Bayesian belief network** represents those dependencies as
a **directed acyclic graph** and stores one conditional probability table per node.

Once the graph is built, the network answers questions in both directions. It can
predict effects from causes, diagnose causes from effects, and handle
**explaining away**: if the sprinkler is already on, wet grass makes rain less
likely than it would otherwise be.

## What You Will Learn

- What a Bayesian belief network is and how it differs from Naive Bayes.
- How to read a directed acyclic graph (DAG) of variables.
- What nodes, edges, parents, children, roots, and leaves mean.
- How conditional probability tables (CPTs) encode local behaviour.
- How the joint distribution factorizes into local CPTs.
- How the graph encodes conditional independence, including chains, forks, and
  colliders.
- What explaining away is and why it requires a network.
- How to perform exact inference by enumeration.
- How Bayesian networks are used in diagnosis, risk, and decision support.
- When exact inference becomes too expensive.

## Detailed Content

- [Theory](theory.md) — DAG representation, CPTs, joint factorization, conditional
  independence, enumeration, explaining away, learning CPTs and structure, a
  comparison with Naive Bayes, applications, strengths, limitations, and common
  mistakes.
- [Practical](practical.md) — build the Rain/Sprinkler/WetGrass network, enumerate
  its eight-row joint distribution, implement exact inference, compute diagnostic
  probabilities, and observe explaining away. An optional `pgmpy` version is
  included.

## Files

No CSV is needed. The practical defines a three-variable network directly in the
notebook:

- `rain` — a root variable with prior 0.2.
- `sprinkler` — a root variable with prior 0.5.
- `wet` — a child of both, with a conditional probability table.

Everything runs with plain Python, NumPy, and pandas. The optional section uses
`pgmpy`, which is listed in `requirements.txt`.

The notebook should be created inside this Day 13 folder.

## Before You Start

- Day 12: conditional probability, Bayes' theorem, priors, and posteriors.
- Day 02: basic probability and distributions.
- Day 04: honest evaluation and the difference between correlation and causation.

## Outcome

Learners can construct a small Bayesian network, write and interpret conditional
probability tables, factorize its joint distribution, compute conditional
probabilities by enumeration, and explain conditional independence and explaining
away with a concrete example.
