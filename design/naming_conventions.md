# Naming Conventions

## Terminology

mlfeatures uses scikit-learn's vocabulary: a column used as model input is a **feature** (never a "predictor"), the column being predicted is the **target** (never an "outcome" or a "label"), and one row is a **sample** (never an "observation"). `"feature"` and `"target"` are the literal role name strings, so the choice is baked into the API surface rather than being a prose convention. The distinct values of a categorical column are its **levels**, never its "labels".

See [`../user_guide/vocabulary.qmd`](../user_guide/vocabulary.qmd) for the full mapping against R `recipes` and scikit-learn, including the false friends, and the Vocabulary section of [`../CLAUDE.md`](../CLAUDE.md) for the rules in checklist form.

## Steps that remove columns

Steps whose primary effect is dropping columns from the dataset must be named
with the prefix `step_drop_`. The corresponding internal fitting helper must
use the prefix `fit_drop_` (not `fit_filter_`).

| Step file | Internal helper |
| -------------------------------- | ------------------------------ |
| `mlfeatures/step_drop_corr.py` | `fit_drop_corr` |

**Rationale:** The `drop` prefix makes the destructive nature of the
transformation unambiguous at a glance, both in user-facing APIs and in the
source tree. The older `filter` prefix was ambiguous (filtering rows vs.
filtering columns) and is no longer used.
