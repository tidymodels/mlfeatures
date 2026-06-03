# Naming Conventions

## Steps that remove columns

Steps whose primary effect is dropping columns from the dataset must be named
with the prefix `step_drop_`. The corresponding internal fitting helper must
use the prefix `fit_drop_` (not `fit_filter_`).

| Step file | Internal helper |
| -------------------------------- | ------------------------------ |
| `shoez/step_drop_corr.py` | `fit_drop_corr` |

**Rationale:** The `drop` prefix makes the destructive nature of the
transformation unambiguous at a glance, both in user-facing APIs and in the
source tree. The older `filter` prefix was ambiguous (filtering rows vs.
filtering columns) and is no longer used.
