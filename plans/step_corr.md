# Plan: `fit_filter_corr` in `shoez/step_corr.py`

## Context

`step_corr` removes columns whose pairwise absolute correlation exceeds a threshold,
porting R's [`recipes::step_corr`](https://github.com/tidymodels/recipes/blob/main/R/corr.R).
This plan covers only the pure fitting function (`fit_filter_corr`) — the statistical
logic that decides which columns to drop — not the full `Step` class.

## Files to Create / Modify

| File | Action |
|---|---|
| `shoez/step_corr.py` | **Create** — contains `fit_filter_corr`; defines `__all__ = ["fit_filter_corr"]` |
| `pyproject.toml` | **Edit** — add `narwhals>=2.19.0`; extend mypy override to include narwhals |
| `tests/test_step_corr.py` | **Create** — unit tests |

`shoez/__init__.py` is **not** touched — `fit_filter_corr` is not a top-level public
export yet; that is the future `Step` class's job.

---

## `pyproject.toml` changes

Add to `[project] dependencies`:

```toml
"narwhals>=2.19.0",
```

Extend the existing mypy override:

```toml
[[tool.mypy.overrides]]
module = ["pandas", "polars", "narwhals", "narwhals.*"]
ignore_missing_imports = true
```

---

## `shoez/step_corr.py` — implementation

### Signature

```python
def fit_filter_corr(
    features: nw.DataFrame,
    threshold: float = 0.9,
    method: Literal["pearson", "spearman"] = "pearson",
) -> npt.NDArray[np.str_]:
```

- `features` — narwhals DataFrame of numeric columns, passed in after selector resolution.
- `threshold` — pairs with |r| **strictly above** this value trigger removal (default 0.9).
- `method` — `"pearson"` or `"spearman"`. Kendall is not supported by `nw.corr()`.
- Returns a numpy array of dtype `np.str_` of column names to drop (empty if nothing to drop).

### Imports

```python
from __future__ import annotations
import warnings
from typing import Literal
import narwhals as nw
import numpy as np
import numpy.typing as npt
```

No scipy, no pandas.

### Algorithm (based on R's `corr_filter`; deviates on zero-variance columns)

1. **Input validation** — raise `ValueError` if `threshold` is not in `[0, 1]` or
   if `method` is not `"pearson"` or `"spearman"`.

2. **Guard** — if `features` has fewer than 2 columns, return an empty array immediately.

3. **Correlation matrix** — loop over all column pairs `(i, j)` where `i < j` and
   call `features.select(nw.corr(col_i, col_j, method=method)).item()` for each pair.
   Build a symmetric `n × n` numpy matrix from the results.
   `nw.corr()` is used as an expression inside `.select()` and delegates
   NaN handling to the underlying backend, giving pairwise-complete-obs semantics
   (equivalent to R's `use = "pairwise.complete.obs"`).

4. **All-NA column detection** — a column is "all-NA" if every off-diagonal entry in
   its correlation matrix column is NaN (zero variance or all-null input).
   - **Add all-NA columns to the drop list immediately** (deviation from R, which
     excludes them from the filter but keeps them). Warn how many were removed.
   - Zero out their rows/columns in the matrix so they don't affect the remaining
     correlation computation.
   - If the number of *remaining* columns (after removing all-NA ones) is fewer than
     2, warn `"Too many correlations are NA; skipping correlation filter."` and return
     the all-NA columns only.
   - Zero out any remaining sporadic NaN and warn.

5. **Fix diagonal** to `1.0`.

6. **Average absolute correlation** per column:
   ```python
   avg_corr = np.mean(np.abs(corr), axis=0)
   ```

7. **Dense rank** (pure numpy, matches R's `as.numeric(as.factor(...))`):
   ```python
   _, inv = np.unique(avg_corr, return_inverse=True)
   ranks = inv + 1  # 1-based
   ```

8. **Upper triangle** — set lower triangle + diagonal to `np.nan` so each pair is
   considered only once.

9. **Find pairs above threshold**:
   ```python
   rows_idx, cols_idx = np.where(np.abs(upper) > threshold)
   ```

10. **Decide which to drop** — for each pair `(r, c)`, drop `c` if
    `ranks[c] > ranks[r]`, else drop `r` (including ties — matches R's behavior
    where equal ranks cause the row variable to be discarded). Collect unique indices.

11. **Return** — combine all-NA drops from step 4 with correlation-based drops from
    step 10, deduplicate, and return as a sorted array:
    ```python
    # np.asarray on an empty list gives a float64 array by default; force
    # the dtype to unicode-string so the result type is stable.
    return np.asarray(result, dtype=np.str_)
    ```

---

## `tests/test_step_corr.py` — cases to cover

| Case | Expected |
|---|---|
| Perfectly correlated pair (r=1.0) | Exactly one column dropped; the one with the higher mean-\|r\| |
| Negatively correlated pair (r≈-1.0) | Column dropped — filter operates on `abs(corr)` |
| Exact removals on a fixed-seed dataset | Assert specific column names, not just count (mirrors R's `high filter` test) |
| Multiple columns — drop the one with highest mean-\|r\| | Correct column dropped |
| No pair above threshold | Empty array |
| Single-column input | Empty array |
| Zero-row input | Empty array, no crash |
| Threshold exactly equal (not strictly greater) | Nothing dropped |
| `threshold > 1` or `threshold < 0` | Raises `ValueError` |
| Invalid `method` string | Raises `ValueError` |
| Zero-variance / all-NaN column | Warning emitted, column is dropped (deviation from R) |
| Too many NaN columns (fewer than 2 non-NA columns remain) | "skipping" warning, returns any all-NA columns already identified |
| Sporadic NaN values | Warning emitted, no crash |
| Return dtype when empty | `dtype.kind == "U"` (unicode string) |
| Return dtype when non-empty | `dtype.kind == "U"` (unicode string) |
| `method="spearman"` | Does not raise |

---

## Verification

```bash
uv run pytest tests/test_step_corr.py -v
uv run ruff check shoez/step_corr.py
uv run ruff format --check shoez/step_corr.py
uv run mypy shoez/step_corr.py
uv run pyright shoez/step_corr.py
```
