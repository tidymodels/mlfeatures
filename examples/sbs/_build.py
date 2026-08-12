#!/usr/bin/env python3
"""Build the line-matched R-vs-Python side-by-side pages in this directory.

The rendered HTML files beside this script are COMMITTED, generated output —
regenerate after editing and commit both:

    python examples/sbs/_build.py

Content lives in the SPECS list below: one spec per example, as sections of
(title, shared_desc_lines, py_diff_lines, r_code, py_code). The code panes are
hand-curated condensations of the research scripts (the R panes carry verbatim
R code and run results that shoez CI cannot reproduce — R runs happened in the
shoez-research repo). The script does the mechanics: shared lines appear
verbatim in both panes; py diff lines get a bare "#" placeholder in R; code
blocks are end-padded to equal length, so both panes stay line-matched (an
assertion enforces it). Underscore-prefixed so Quarto never renders it.
"""
import html, os

ROOT = os.path.dirname(os.path.abspath(__file__))
W = 76

def banner(t):
    return f"# {t} " + "-" * max(4, W - len(t) - 3)

def build_panes(sections):
    r_lines, py_lines = [], []
    for title, shared, diffs, r_code, py_code in sections:
        r_lines.append(banner(title)); py_lines.append(banner(title))
        for t in shared:
            r_lines.append(f"# {t}"); py_lines.append(f"# {t}")
        for t in diffs:
            r_lines.append("#"); py_lines.append(f"# {t}")
        r_lines.append(""); py_lines.append("")
        rc, pc = r_code.splitlines(), py_code.splitlines()
        n = max(len(rc), len(pc))
        r_lines += rc + [""] * (n - len(rc))
        py_lines += pc + [""] * (n - len(pc))
        r_lines.append(""); py_lines.append("")
    assert len(r_lines) == len(py_lines)
    return r_lines, py_lines

def render(lines):
    out, mode = [], None
    for l in lines:
        e = html.escape(l)
        if "----" in l and l.startswith("# "):
            out.append(f'<span class="bn">{e}</span>'); mode = None
        elif l.startswith("# DIFF (GAP") or l.startswith("# DIFF (shoez-only"):
            out.append(f'<span class="gp">{e}</span>'); mode = "gp"
        elif l.startswith("# DIFF"):
            out.append(f'<span class="df">{e}</span>'); mode = "df"
        elif l.startswith("# ") and mode in ("gp", "df"):
            out.append(f'<span class="{mode}">{e}</span>')
        elif l.startswith("#"):
            out.append(f'<span class="cm">{e}</span>'); mode = None
        else:
            out.append(e); mode = None
    return "\n".join(out)

CSS = """
:root {
  --paper:#FBFAF6; --ink:#26241F; --muted:#6E695E; --line:#E2DFD4; --card:#F3F1E9;
  --accent:#A05E12; --blue:#41628A; --gap:#8A2F1D;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --serif:"Iowan Old Style",Palatino,Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme: dark) { :root {
  --paper:#16171A; --ink:#E8E6DF; --muted:#9B978C; --line:#2D2F35; --card:#1E2025;
  --accent:#D89A4E; --blue:#83A7CF; --gap:#E09383;
} }
html { background:var(--paper); }
body { font-family:var(--sans); color:var(--ink); margin:0; padding:2rem 1.25rem 4rem; }
.wrap { max-width:110rem; margin:0 auto; }
h1 { font-family:var(--serif); font-size:1.5rem; font-weight:600; margin:0 0 .2rem; }
.lede { color:var(--muted); font-size:.85rem; margin:0 0 1.2rem; }
.nav { font-size:.78rem; font-family:var(--mono); margin:0 0 .8rem; }
.nav a { color:var(--blue); text-decoration:none; margin-right:.7rem; }
.nav a.here { color:var(--accent); font-weight:700; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.pane h2 { font-family:var(--mono); font-size:.8rem; margin:0 0 .4rem; }
.pane.r h2 { color:var(--accent); }
.pane.py h2 { color:var(--blue); }
pre { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:0;
      padding:.8rem 1rem; overflow-x:auto; font-family:var(--mono);
      font-size:.72rem; line-height:1.5; }
.bn { font-weight:700; }
.cm { color:var(--muted); }
.df { color:var(--accent); }
.gp { color:var(--gap); font-weight:600; }
ul.idx { line-height:1.9; }
@media (max-width: 70rem) { .cols { grid-template-columns:1fr; } }
"""

# =============================================================================
# Specs. Codes are lightly condensed from candidates/<id>/example.{R,py}.
# =============================================================================

SPECS = [
# ---------------------------------------------------------------- TMWR-01 ---
dict(slug="tmwr01", id="TMWR-01", title="ames-mini + lm",
     status="R: rmse 0.0764 / rsq 0.824 · Python ran: 0.0825 / 0.771 — same recipe, different split rows (seeds can't transfer). Identical metrics on pandas and polars backends.",
     sections=[
("Setup",
 ["Load the modeling stack."],
 ["DIFF: tidymodels bundles everything; Python assembles polars + sklearn + shoez."],
 """library(tidymodels)
tidymodels_prefer()""",
 """import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from shoez import Recipe, RecipeTransformer, StepDummy
from shoez import Selectors as s"""),
("Data + outcome transform",
 ["Four predictors; log10 the outcome BEFORE the recipe."],
 ["DIFF: ames exported once from R (no PyPI package) -> data/ames.csv;",
  "polars needs null_values='NA' — R's NA strings aren't auto-parsed.",
  "DIFF (GAP): step_log has no shoez analog — the log moves PRE-recipe.",
  "Static, so safe, but it leaves the replay boundary: new frames must",
  "arrive already-logged. That trade IS the escape-hatch lesson."],
 """data(ames, package = "modeldata")
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))

# (Gr_Liv_Area is logged INSIDE the recipe, below)""",
 """ames = pl.read_csv("data/ames.csv", null_values="NA")
ames = ames.select(["Neighborhood", "Gr_Liv_Area", "Year_Built",
                    "Bldg_Type", "Sale_Price"])
ames = ames.with_columns(pl.col("Sale_Price").log(base=10),
                         pl.col("Gr_Liv_Area").log(base=10))  # was step_log"""),
("Split",
 ["Stratified 80/20 split on the outcome."],
 ["DIFF: explicit quartile binning (qcut); seeds don't transfer across",
  "languages — same design, different rows."],
 """set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)""",
 """strata = ames["Sale_Price"].qcut(4, labels=["q1","q2","q3","q4"])
ames_train, ames_test = train_test_split(
    ames, train_size=0.80, stratify=strata, random_state=502
)"""),
("Recipe",
 ["The book's first recipe: the formula's implicit preprocessing made explicit."],
 ["DIFF: with the log moved out, only the encoding remains in the recipe.",
  "DIFF: StepDummy = step_dummy (k-1 indicators, reference level dropped)."],
 """simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area +
           Year_Built + Bldg_Type, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_dummy(all_nominal_predictors())""",
 """simple_ames = (
    Recipe(data=ames_train, outcome="Sale_Price")
    .add(StepDummy(s.nominal_predictors()))
)"""),
("Model + fit",
 ["Bind recipe + lm; fit on the training set."],
 ["DIFF: Pipeline + RecipeTransformer = workflow(); the X/y split is sklearn",
  "idiom — RecipeTransformer re-joins y as the outcome internally."],
 """lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(simple_ames)

lm_fit <- fit(lm_wflow, ames_train)""",
 """lm_wflow = Pipeline([
    ("recipe", RecipeTransformer(simple_ames)),
    ("model", LinearRegression()),
])

X_train = ames_train.drop("Sale_Price")
y_train = ames_train["Sale_Price"]
lm_wflow.fit(X_train, y_train)"""),
("Proof + holdout metrics",
 ["Bake/transform the test set; score once."],
 ["DIFF: metric_set(rmse, rsq) -> polars expressions. Fun finding: y stays a",
  "polars Series through ALL of sklearn (fit, CV, predict, metrics) — the one",
  "thing that broke was np.mean() on a polars Series (numpy passes axis=)."],
 """extract_recipe(lm_fit, estimated = TRUE) %>%
  bake(new_data = ames_test) %>%
  head()

predict(lm_fit, ames_test) %>%
  bind_cols(ames_test %>% select(Sale_Price)) %>%
  reg_metrics(truth = Sale_Price, estimate = .pred)
# run result: rmse 0.0764, rsq 0.824 (log10 units)""",
 """simple_ames.fit(ames_train).transform(ames_test).head()

pred = lm_wflow.predict(X_test)
rmse = float(((y_test - pred) ** 2).mean() ** 0.5)
rsq  = float(pl.DataFrame({"y": y_test, "pred": pred})
             .select(pl.corr("y", "pred")).item() ** 2)


# ran: rmse 0.0825, rsq 0.771 (R: 0.0764/0.824 — split-row luck)"""),
]),

# ---------------------------------------------------------------- TMWR-03 ---
dict(slug="tmwr03", id="TMWR-03", title="ames realistic + lm",
     status="R: CV rmse 0.0791 · Python ran: test 0.0825, CV = nan — unseen fold-levels nulled predictions; the gaps broke CV itself (see py-run-results.md).",
     sections=[
("Setup",
 ["Load the modeling stack."],
 ["DIFF: adds cross-validation imports; otherwise as TMWR-01."],
 """library(tidymodels)
tidymodels_prefer()""",
 """import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from shoez import Recipe, RecipeTransformer, StepDummy
from shoez import Selectors as s"""),
("Data + outcome transform",
 ["Six predictors now (adds Latitude/Longitude); log10 the outcome pre-split."],
 ["DIFF (GAP): step_log again worked around pre-recipe (see TMWR-01)."],
 """data(ames, package = "modeldata")
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))""",
 """ames = pl.read_csv("data/ames.csv", null_values="NA")
ames = ames.select(["Neighborhood", "Gr_Liv_Area", "Year_Built", "Bldg_Type",
                    "Latitude", "Longitude", "Sale_Price"])
ames = ames.with_columns(pl.col("Sale_Price").log(base=10),
                         pl.col("Gr_Liv_Area").log(base=10))  # was step_log"""),
("Split",
 ["Stratified 80/20 split on the outcome (same as TMWR-01)."],
 [],
 """set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)""",
 """strata = ames["Sale_Price"].qcut(4, labels=["q1","q2","q3","q4"])
ames_train, ames_test = train_test_split(
    ames, train_size=0.80, stratify=strata, random_state=502
)"""),
("Recipe",
 ["The canonical recipe. Ordering is load-bearing twice: other pools rare",
  "levels BEFORE dummy; interact targets the GENERATED Bldg_Type_* columns."],
 ["DIFF (GAP): step_other unbuilt — rare Neighborhoods keep their own dummies,",
  "so the design matrix is wider and sparser than R's.",
  "DIFF (GAP): step_interact unbuilt — interactions over generated columns is",
  "exactly shoez's has_source pitch; target form shown in place.",
  "DIFF (GAP): step_ns unbuilt — no spline steps at all."],
 """ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built +
           Bldg_Type + Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)""",
 """ames_rec = (
    Recipe(data=ames_train, outcome="Sale_Price")

    # .add(StepOther(s.col("Neighborhood"), threshold=0.01))   # GAP
    .add(StepDummy(s.nominal_predictors()))
    # .add(StepInteract(s.col("Gr_Liv_Area")
    #                   * s.starts_with("Bldg_Type_")))         # GAP
    # .add(StepSplineNatural(s.col("Latitude", "Longitude"),
    #                        deg_free=20))                      # GAP
)"""),
("Resampled performance",
 ["5-fold CV estimate of the whole pipeline (recipe re-fit per fold)."],
 ["DIFF: fit_resamples -> cross_val_score over the Pipeline — both re-estimate",
  "the recipe inside each fold, so there's no leakage in either language."],
 """set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 5)

lm_res <- fit_resamples(lm_wflow, resamples = ames_folds)
collect_metrics(lm_res)
# run result: CV rmse 0.0791, rsq 0.798""",
 """cv = KFold(n_splits=5, shuffle=True, random_state=1001)

cv_rmse = -cross_val_score(lm_wflow, X_train, y_train, cv=cv,
                           scoring="neg_root_mean_squared_error")
print(cv_rmse.mean())
# ran: nan! unseen fold-level ('Landmark') nulled predictions — the
# missing step_other doesn't just cost accuracy, it breaks CV"""),
("Final fit + proof + metrics",
 ["Fit on all of training; transform the test set; score once."],
 [],
 """lm_fit <- fit(lm_wflow, ames_train)

extract_recipe(lm_fit, estimated = TRUE) %>%
  bake(new_data = ames_test) %>% head()

predict(lm_fit, ames_test) %>%
  bind_cols(ames_test %>% select(Sale_Price)) %>%
  reg_metrics(truth = Sale_Price, estimate = .pred)""",
 """lm_wflow.fit(X_train, y_train)

ames_rec.fit(ames_train).transform(ames_test).head()

pred = lm_wflow.predict(X_test)
rmse = float(((y_test - pred) ** 2).mean() ** 0.5)
rsq  = float(pl.DataFrame({"y": y_test, "pred": pred})
             .select(pl.corr("y", "pred")).item() ** 2)"""),
]),

# ---------------------------------------------------------------- TMWR-11 ---
dict(slug="tmwr11", id="TMWR-11", title="ames + xgboost",
     status="R: test rmse 0.0730 (best CV 0.0773) · Python ran: test 0.0771 / rsq 0.800.",
     sections=[
("Setup",
 ["Load the modeling stack."],
 ["DIFF: tidymodels is one meta-package; Python assembles polars + sklearn",
  "+ xgboost + shoez (xgboost's polars path needs pyarrow)."],
 """library(tidymodels)
tidymodels_prefer()""",
 """import numpy as np
import polars as pl
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from shoez import Recipe, RecipeTransformer, StepDropZeroVariance, StepOneHot
from shoez import Selectors as s"""),
("Data + outcome transform",
 ["Ames housing; log10 the outcome BEFORE the recipe (outcome transforms",
  "don't belong inside it — they fail at predict time)."],
 ["DIFF: no ames on PyPI — exported once from R: modeldata::ames -> data/ames.csv.",
  "DIFF: the CSV drops R's factor dtypes; plain strings are fine, because",
  "nominal_predictors() matches them (categorical_predictors() would not)."],
 """data(ames, package = "modeldata")
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))

# (predictors declared in the recipe formula, below)""",
 """ames = pl.read_csv("data/ames.csv", null_values="NA")
ames = ames.select(["Neighborhood", "Gr_Liv_Area", "Year_Built",
                    "Bldg_Type", "Latitude", "Longitude", "Sale_Price"])
ames = ames.with_columns(pl.col("Sale_Price").log(base=10))"""),
("Split",
 ["Stratified 80/20 split on the outcome."],
 ["DIFF: explicit quartile binning (qcut) does what initial_split(strata=)",
  "does internally.",
  "DIFF: seeds don't transfer across languages — same design, different rows."],
 """set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)""",
 """strata = ames["Sale_Price"].qcut(4, labels=["q1","q2","q3","q4"])
ames_train, ames_test = train_test_split(
    ames, train_size=0.80, stratify=strata, random_state=502
)"""),
("Recipe",
 ["The usemodels \"minimal preprocessing for xgboost\": guard unseen levels,",
  "one-hot encode, drop zero-variance columns."],
 ["DIFF (GAP): step_novel() has no shoez analog yet — StepNovel is unbuilt and",
  "StepUnknown fills only nulls. Omitted here; StepOneHot warns and emits",
  "nulls for unseen levels at transform instead.",
  "DIFF: Recipe(outcome=...) roles replace the R formula; selectors pick",
  "columns by kind."],
 """xgboost_recipe <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built +
           Bldg_Type + Latitude + Longitude, data = ames_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())""",
 """xgboost_recipe = (
    Recipe(data=ames_train, outcome="Sale_Price")
    .add(StepOneHot(s.nominal_predictors()))
    .add(StepDropZeroVariance(s.predictors()))
)"""),
("Model spec + workflow",
 ["Bind recipe + model; trees and learn_rate will be tuned."],
 ["DIFF: Pipeline + RecipeTransformer = workflow(); the tune() markers move",
  "into param_grid (next section) — sklearn declares tunables at search time."],
 """xgboost_spec <-
  boost_tree(trees = tune(), learn_rate = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

xgboost_workflow <-
  workflow() %>%
  add_recipe(xgboost_recipe) %>%
  add_model(xgboost_spec)""",
 """pipe = Pipeline([
    ("recipe", RecipeTransformer(xgboost_recipe)),
    ("model", XGBRegressor(objective="reg:squarederror",
                           random_state=69305)),
])"""),
("Resamples + grid + tune",
 ["5-fold CV over the same 2 x 3 grid (trees x learn_rate)."],
 ["DIFF: GridSearchCV is three tidymodels verbs in one object:",
  "tune_grid (search) + select_best (pick) + refit on full train (refit=True).",
  "DIFF: sklearn separates X from y; RecipeTransformer re-joins y as the",
  "outcome internally, so the recipe still sees one frame."],
 """set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 5)

xgb_grid <- crossing(
  trees      = c(500, 1000),
  learn_rate = c(0.01, 0.05, 0.1)
)

set.seed(69305)
xgboost_tune <- tune_grid(xgboost_workflow,
                          resamples = ames_folds,
                          grid = xgb_grid)
show_best(xgboost_tune, metric = "rmse", n = 3)""",
 """cv = KFold(n_splits=5, shuffle=True, random_state=1001)

param_grid = {
    "model__n_estimators": [500, 1000],
    "model__learning_rate": [0.01, 0.05, 0.1],
}

X_train = ames_train.drop("Sale_Price")
y_train = ames_train["Sale_Price"]
search = GridSearchCV(pipe, param_grid, cv=cv,
                      scoring="neg_root_mean_squared_error",
                      refit=True)
search.fit(X_train, y_train)"""),
("Finalize + test metrics",
 ["Refit the best config on all training data; evaluate once on the test set."],
 ["DIFF: last_fit() is one verb for refit-and-evaluate; the refit already",
  "happened (refit=True), so only the scoring remains."],
 """best_xgb  <- select_best(xgboost_tune, metric = "rmse")
final_res <- xgboost_workflow %>%
  finalize_workflow(best_xgb) %>%
  last_fit(ames_split)
collect_metrics(final_res)  # run result: test rmse 0.0773 (log10)""",
 """X_test = ames_test.drop("Sale_Price")
y_test = ames_test["Sale_Price"]
pred  = search.predict(X_test)
rmse  = float(((y_test - pred) ** 2).mean() ** 0.5)
print(rmse)                 # ran: 0.0771 (R test: 0.0730)"""),
("Proof the recipe stands alone",
 ["Transform new data with no model attached."],
 ["DIFF (shoez-only): the lineage query — which columns did the one-hot step",
  "produce? — has no recipes equivalent (name-prefix matching is the R workaround)."],
 """final_res %>%
  extract_recipe(estimated = TRUE) %>%
  bake(new_data = ames_test) %>%
  head()""",
 """fitted = xgboost_recipe.fit(ames_train)
fitted.transform(ames_train).head()

fitted.schema.columns_from_source("one_hot")[:8]"""),
]),

# ---------------------------------------------------------------- TMWR-14 ---
dict(slug="tmwr14", id="TMWR-14", title="concrete + KNN",
     status="R: test rmse 8.57 · Python ran: 7.94 — ports 1:1 and ran clean; the dual-ecosystem candidate, confirmed.",
     sections=[
("Setup",
 ["Load the modeling stack."],
 [],
 """library(tidymodels)
tidymodels_prefer()""",
 """import numpy as np
import polars as pl
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from shoez import Recipe, RecipeTransformer, StepNormalize
from shoez import Selectors as s"""),
("Data + the leakage beat",
 ["Average replicate mixtures BEFORE splitting, so one mixture can't appear",
  "in both train and test. Data-layer work, pre-recipe, in both languages."],
 ["DIFF: dplyr group_by/summarize -> polars group_by/agg. Two polars gotchas:",
  "infer_schema_length (int-looking early rows lock Int64, then floats fail)",
  "and group_by's unordered output — the sort restores determinism."],
 """data(concrete, package = "modeldata")

concrete <- concrete %>%
  group_by(across(-compressive_strength)) %>%
  summarize(compressive_strength = mean(compressive_strength),
            .groups = "drop")   # 1,030 rows -> 992""",
 """concrete = pl.read_csv("data/concrete.csv",
                       infer_schema_length=None)
ingredients = [c for c in concrete.columns
               if c != "compressive_strength"]
concrete = (concrete.group_by(ingredients)
            .agg(pl.col("compressive_strength").mean())
            .sort(ingredients))   # 1,030 -> 992"""),
("Split",
 ["Stratified 75/25 split on the outcome."],
 [],
 """set.seed(1501)
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)""",
 """strata = (concrete["compressive_strength"]
          .qcut(4, labels=["q1","q2","q3","q4"]))
concrete_train, concrete_test = train_test_split(
    concrete, train_size=0.75, stratify=strata, random_state=1501
)"""),
("Recipe",
 ["One step. \"KNN ... require predictors that have been centered and scaled.\""],
 ["DIFF: none — StepNormalize ports step_normalize exactly. This candidate",
  "runs in shoez today.",
  "DIFF (GAP, variant only): the poly/interact response surface needs the",
  "all-pairs StepInteract (unbuilt); StepPolynomial itself exists."],
 """normalized_rec <-
  recipe(compressive_strength ~ ., data = concrete_train) %>%
  step_normalize(all_predictors())

# poly_recipe <- normalized_rec %>%
#    step_poly(all_predictors()) %>%
#    step_interact(~ all_predictors():all_predictors())""",
 """normalized_rec = (
    Recipe(data=concrete_train, outcome="compressive_strength")
    .add(StepNormalize(s.predictors()))
)
# poly_recipe = normalized_rec.add(
#     StepPolynomial(s.predictors(), degree=2))
#     # .add(StepInteract(s.predictors() * s.predictors()))  # GAP"""),
("Model + tune",
 ["Tune the neighborhood over 5-fold CV."],
 ["DIFF: kknn tunes neighbors / dist_power / weight_func (kernel weights);",
  "sklearn's analogs are n_neighbors / p / weights (only uniform-vs-distance)",
  "— same idea, coarser space, so best configs won't match exactly."],
 """knn_spec <-
  nearest_neighbor(neighbors = tune(), dist_power = tune(),
                   weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_wflow <- workflow() %>%
  add_recipe(normalized_rec) %>%
  add_model(knn_spec)

set.seed(1503)
knn_tune <- tune_grid(knn_wflow, resamples = concrete_folds,
                      grid = 6)
show_best(knn_tune, metric = "rmse", n = 3)""",
 """pipe = Pipeline([
    ("recipe", RecipeTransformer(normalized_rec)),
    ("model", KNeighborsRegressor()),
])

param_grid = {
    "model__n_neighbors": [4, 8, 12, 16],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}
cv = KFold(n_splits=5, shuffle=True, random_state=1502)
search = GridSearchCV(pipe, param_grid, cv=cv,
                      scoring="neg_root_mean_squared_error", refit=True)
search.fit(X_train, y_train)"""),
("Finalize + proof + metrics",
 ["Refit the best config; transform the test set; score once."],
 [],
 """best_knn  <- select_best(knn_tune, metric = "rmse")
final_res <- knn_wflow %>%
  finalize_workflow(best_knn) %>%
  last_fit(concrete_split)

final_res %>% extract_recipe(estimated = TRUE) %>%
  bake(new_data = concrete_test) %>% head()
collect_metrics(final_res)   # run: CV rmse ~ 9.47 MPa""",
 """# refit already happened (refit=True)
fitted = normalized_rec.fit(concrete_train)
fitted.transform(concrete_test).head()

pred = search.predict(X_test)
rmse = float(((y_test - pred) ** 2).mean() ** 0.5)
print(rmse)                  # ran: 7.94 (R test: 8.57 — distance weights won)"""),
]),

# ---------------------------------------------------------------- TMWR-16 ---
dict(slug="tmwr16", id="TMWR-16", title="beans PCA, no model",
     status="Both ran. StepPCA still unbuilt — sklearn PCA stands in outside the recipe; without orderNorm, PC1-4 variance = 55.4/26.5/8.0/5.1%.",
     sections=[
("Setup",
 ["Load the stack. No estimator anywhere — the recipe is the product."],
 ["DIFF: sklearn appears only for PCA (the unbuilt step), not for a model."],
 """library(tidymodels)
tidymodels_prefer()
library(beans)
library(bestNormalize)""",
 """import polars as pl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from shoez import Recipe, StepDropZeroVariance, StepNormalize
from shoez import Selectors as s"""),
("3-way split",
 ["75 / 12.5 / 12.5 train / validation / test, stratified by class —",
  "extraction is judged on the validation set, honestly."],
 ["DIFF: initial_validation_split -> two chained train_test_splits."],
 """set.seed(1601)
bean_split <- initial_validation_split(beans, strata = class,
                                       prop = c(0.75, 0.125))
bean_train      <- training(bean_split)
bean_validation <- validation(bean_split)
bean_test       <- testing(bean_split)""",
 """bean_train, rest = train_test_split(
    beans, train_size=0.75, stratify=beans["class"],
    random_state=1601)
bean_validation, bean_test = train_test_split(
    rest, train_size=0.50, stratify=rest["class"],
    random_state=1601)"""),
("Base recipe",
 ["zv guard, tame the skew, shared scale — PCA's preconditions generate",
  "the step ordering. fit() ~ prep(); transform() ~ bake()."],
 ["DIFF (GAP): step_orderNorm (bestNormalize's ORQ transform) has no shoez",
  "analog — not even designed. Omitted: skewed shape features stay skewed,",
  "so components will differ somewhat from the book's."],
 """bean_rec <-
  recipe(class ~ ., data = bean_train) %>%
  step_zv(all_numeric_predictors()) %>%
  step_orderNorm(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

bean_rec_trained <- prep(bean_rec)
bake(bean_rec_trained, new_data = bean_validation) %>% head()""",
 """bean_rec = (
    Recipe(data=bean_train, outcome="class")
    .add(StepDropZeroVariance(s.numeric_predictors()))

    .add(StepNormalize(s.numeric_predictors()))
)
fitted = bean_rec.fit(bean_train)
fitted.transform(bean_validation).head()"""),
("PCA",
 ["Four components, estimated on training data only, scored on validation."],
 ["DIFF (GAP): StepPCA is designed-but-unbuilt — sklearn PCA consumes the",
  "recipe's output OUTSIDE the replay boundary. When it lands, this block",
  "collapses into .add(StepPCA(s.numeric_predictors(), num_comp=4)).",
  "DIFF: R pipes a NEW step onto the already-prep()ed recipe and re-preps",
  "(incremental fit); shoez recipes are declared whole and fit once."],
 """bean_pca_rec <-
  bean_rec_trained %>%
  step_pca(all_numeric_predictors(), num_comp = 4, id = "pca") %>%
  prep()

bake(bean_pca_rec, new_data = bean_validation) %>% head()""",
 """pca = PCA(n_components=4, random_state=1601)
pca.fit(fitted.transform(bean_train).select(num_cols))

scores = pl.DataFrame(
    pca.transform(fitted.transform(bean_validation).select(num_cols)),
    schema=["PC1", "PC2", "PC3", "PC4"])
scores.head()"""),
("Loadings + variance",
 ["The no-model deliverables: top loadings and percent variance."],
 ["DIFF: tidy(id = \"pca\") -> pca.components_ / explained_variance_ratio_.",
  "A fitted StepPCA should expose the same queries on the recipe object."],
 """tidy(bean_pca_rec, id = "pca") %>%
  group_by(component) %>%
  slice_max(abs(value), n = 5)

tidy(bean_pca_rec, id = "pca", type = "variance")""",
 """loadings = pl.DataFrame({"term": num_cols}).with_columns(
    pl.Series(f"PC{i+1}", pca.components_[i]) for i in range(4))
loadings.unpivot(index="term").sort("value", descending=True)

print(pca.explained_variance_ratio_ * 100)"""),
]),

# ------------------------------------------------------------------- KUHN ---
dict(slug="kuhn", id="KUHN", title="credit_data + glmnet logistic",
     status="R: roc_auc 0.844 / acc 0.806 · Python ran: 0.829 / 0.796 — imputation spine ported 1:1; only step_other missing.",
     sections=[
("Setup",
 ["Load the modeling stack."],
 [],
 """library(tidymodels)""",
 """import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from shoez import (Recipe, RecipeTransformer, StepDropZeroVariance,
                   StepDummy, StepImputeMedian, StepImputeMode,
                   StepNormalize)
from shoez import Selectors as s"""),
("Data + split",
 ["credit_data: 4,454 x 14 with NAs in several predictors — the imputation",
  "example no polished resource provides."],
 ["DIFF: exported from R -> data/credit.csv. null_values='NA' is load-bearing",
  "here — the dataset's whole point is its NAs, and polars would read them",
  "as the string 'NA' (pandas auto-parses them).",
  "DIFF: sklearn wants a numeric target; Status maps to bad=1. R scores the",
  "first factor level (\"bad\") — same event, made explicit here."],
 """data(credit_data, package = "modeldata")

set.seed(55)
credit_split <- initial_split(credit_data, strata = Status)
credit_train <- training(credit_split)
credit_test  <- testing(credit_split)""",
 """credit = pl.read_csv("data/credit.csv", null_values="NA")
credit = credit.with_columns(
    (pl.col("Status") == "bad").cast(pl.Int8).alias("Status"))
credit_train, credit_test = train_test_split(
    credit, train_size=0.75, stratify=credit["Status"],
    random_state=55)"""),
("Recipe",
 ["impute -> collapse rare levels -> dummy -> filter -> normalize."],
 ["DIFF (GAP): StepOther unbuilt — omitted; rare levels keep their dummies.",
  "The R run surfaced the design input: credit_data's Home column already",
  "HAS a level named \"other\", colliding with step_other's default label.",
  "StepOther needs a collision strategy, not just a default string.",
  "DIFF: everything else ports 1:1 — the imputation spine runs today."],
 """credit_rec <-
  recipe(Status ~ ., data = credit_train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05,
             other = "infrequent") %>%  # dodges the "other" collision
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())""",
 """credit_rec = (
    Recipe(data=credit_train, outcome="Status")
    .add(StepImputeMedian(s.numeric_predictors()))
    .add(StepImputeMode(s.nominal_predictors()))

    # .add(StepOther(s.nominal_predictors(), threshold=0.05))  # GAP
    .add(StepDummy(s.nominal_predictors()))
    .add(StepDropZeroVariance(s.predictors()))
    .add(StepNormalize(s.numeric_predictors()))
)"""),
("Model + penalty tuning",
 ["Lasso-penalized logistic; tune the penalty on stratified 5-fold CV."],
 ["DIFF: sklearn 1.8+ speaks glmnet: l1_ratio IS mixture (1.0 = lasso).",
  "DIFF: the glmnet mapping is C = 1/(n*lambda), NOT 1/lambda — glmnet",
  "minimizes MEAN loss + lambda*penalty, sklearn C*TOTAL loss + penalty.",
  "A first draft used 1/lambda; the giveaway was best-C pinning to the",
  "grid edge. Worth a doc callout — easy to get subtly wrong."],
 """glmnet_spec <-
  logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

credit_wflow <- workflow(credit_rec, glmnet_spec)

set.seed(4943)
credit_folds <- vfold_cv(credit_train, v = 5, strata = Status)
penalty_grid <- grid_regular(penalty(range = c(-5, -1)),
                             levels = 20)
credit_res <- tune_grid(credit_wflow, resamples = credit_folds,
                        grid = penalty_grid,
                        metrics = metric_set(roc_auc))""",
 """pipe = Pipeline([
    ("recipe", RecipeTransformer(credit_rec)),
    ("model", LogisticRegression(l1_ratio=1.0, solver="liblinear",
                                 max_iter=5000)),
])

lambdas = np.logspace(-5, -1, 20)
param_grid = {"model__C": 1 / (credit_train.height * lambdas)}
cv = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=4943)
search = GridSearchCV(pipe, param_grid, cv=cv,
                      scoring="roc_auc", refit=True)
search.fit(X_train, y_train)"""),
("Finalize + proof + metrics",
 ["Refit the best penalty; transform the training head; score the test set."],
 [],
 """credit_best <- select_best(credit_res, metric = "roc_auc")
final_res <- credit_wflow %>%
  finalize_workflow(credit_best) %>%
  last_fit(credit_split, metrics = metric_set(roc_auc, accuracy))

prep(credit_rec, training = credit_train) %>%
  bake(new_data = NULL) %>% head()
collect_metrics(final_res)  # run: roc_auc 0.844, accuracy 0.806""",
 """# refit already happened (refit=True)
fitted = credit_rec.fit(credit_train)
fitted.transform(credit_train).head()

prob = search.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob),
      accuracy_score(y_test, search.predict(X_test)))
# ran: roc_auc 0.829, accuracy 0.796 (R: 0.844 / 0.806)"""),
]),
]

# =============================================================================

NAV = " ".join(f'<a href="{sp["slug"]}-r-vs-py.html">{sp["id"]}</a>' for sp in SPECS)

for sp in SPECS:
    r_lines, py_lines = build_panes(sp["sections"])
    nav = NAV.replace(f'<a href="{sp["slug"]}-r-vs-py.html">{sp["id"]}</a>',
                      f'<a class="here" href="{sp["slug"]}-r-vs-py.html">{sp["id"]}</a>')
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sp['id']}: R vs Python, line-matched</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>{sp['id']} · {sp['title']} — R vs Python, line-matched</h1>
<p class="nav"><a href="sbs-index.html">index</a> · {nav}</p>
<p class="lede">Identical section banners and shared descriptions in both scripts;
<span style="color:var(--accent)">ochre DIFF comments</span> (Python side only, R holds a blank <code>#</code>)
explain what changes; <span style="color:var(--gap)">red = shoez gap</span>.
Lightly condensed from <code>candidates/</code> — {sp['status']}</p>
<div class="cols">
<div class="pane r"><h2>example.R — tidymodels (run)</h2>
<pre>{render(r_lines)}</pre></div>
<div class="pane py"><h2>example.py — shoez + sklearn (run)</h2>
<pre>{render(py_lines)}</pre></div>
</div>
</div>
</body>
</html>"""
    out = os.path.join(ROOT, f"{sp['slug']}-r-vs-py.html")
    open(out, "w").write(page)
    print(f"{sp['slug']}: {len(r_lines)} lines per pane")

index = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>R vs Python side-by-sides — proposed six</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>R vs Python, line-matched — the proposed six</h1>
<p class="lede">One page per proposed example: the R script (run) and its shoez/sklearn
port (draft) side by side, sections aligned, differences as in-script comments.
Generated by <code>build-sbs.py</code>.</p>
<ul class="idx">
{"".join(f'<li><a href="{sp["slug"]}-r-vs-py.html">{sp["id"]} · {sp["title"]}</a> — {sp["status"]}</li>' for sp in SPECS)}
</ul>
</div>
</body>
</html>"""
open(os.path.join(ROOT, "sbs-index.html"), "w").write(index)
print("sbs-index.html written")
