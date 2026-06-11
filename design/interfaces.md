# Construction Interfaces

→ See also: [lifecycle.md](lifecycle.md), [extensibility.md](extensibility.md), [step-contract.md](step-contract.md), [goals.md § open questions](goals.md)

---

This page works through the candidate **construction interfaces** — the spelling a user types to declare a sequence of steps. It lays out the live options, weighs each against the project's stated goals, and recommends one.

---

## The two constraints that decide this

Two of the framework's stated goals constrain the surface syntax more than anything else:

1. Steps are independent, first-class objects.
2. Third parties add steps without modifying core.

---

## Option A — Step objects

Step constructors produce plain `Step` **instances**. The recipe is assembled from those objects — either passed as a list at construction, or appended with a builder method.

```python
from shoez import Recipe
from shoez.steps import ImputeMedian, Dummy, Normalize

rec = Recipe(
    formula="price ~ .",
    data=df_train,
    steps=[
        ImputeMedian(all_numeric_predictors()),
        Dummy(all_nominal_predictors()),
        Normalize(all_numeric_predictors()),
    ],
)
```

Equivalently, an **immutable builder** that returns a new recipe each call (note: unlike the gist's mutating `rec.add(...)`, this returns rather than mutates, preserving the purity guarantee from [lifecycle.md](lifecycle.md)):

```python
rec = (
    Recipe(formula="price ~ .", data=df_train)
    .add(ImputeMedian(all_numeric_predictors()))
    .add(Dummy(all_nominal_predictors()))
    .add(Normalize(all_numeric_predictors()))
)
# or variadic: .add(step1, step2, step3)
```

---

## Option B — Operator pipe (`>>`)

Steps are objects (as in A), composed with an overloaded operator.

```python
rec = (
    Recipe(formula="price ~ .", data=df_train)
    >> ImputeMedian(all_numeric_predictors())
    >> Dummy(all_nominal_predictors())
    >> Normalize(all_numeric_predictors())
)
```
