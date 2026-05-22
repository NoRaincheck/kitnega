# merlin

Machine learning library using random-split trees. Stdlib + `lib.array` only.
All three forest types share the same underlying `_Node` tree structure and are
unified under a single `RandomForest(task=…, split=…)` class.

## Usage

```python
from lib.array import array
from merlin.forest import RandomForest
from merlin.convert import extra_to_mondrian, mondrian_to_extra, extra_to_iso

X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = array([0.0, 0.0, 1.0, 1.0])

# Unified API: pick a split strategy and task
clf = RandomForest(n_estimators=10, task="classifier", split="best", random_state=42)
clf.fit(X, y)
clf.predict(X)               # array([0., 0., 1., 1.])
clf.predict_proba(X)         # shape (n, n_classes)

# Regression
reg = RandomForest(n_estimators=10, task="regressor", split="best", random_state=42)
reg.fit(X, array([1.0, 2.0, 3.0, 4.0]))
reg.predict(X)

# Isolation forest — anomaly detection
iso = RandomForest(n_estimators=100, task="anomaly", split="random", random_state=42)
iso.fit(X)
iso.score_samples(X)         # anomaly scores (higher = more normal)
iso.predict(X)               # -1 = outlier, 1 = inlier

# Mondrian forest — online learning
mf = RandomForest(n_estimators=10, task="classifier", split="mondrian", random_state=42)
mf.partial_fit(X[:2], y[:2], classes=[0.0, 1.0])
mf.partial_fit(X[2:], y[2:])
mf.predict(X)

# Convert between forest types (same shared node structure)
mf2 = extra_to_mondrian(clf)          # ExtraForest → MondrianForest
ef2 = mondrian_to_extra(mf)           # MondrianForest → ExtraForest
iso2 = extra_to_iso(clf)              # ExtraForest → IsolationForest
```

Backward-compatible wrappers are also available at the same import paths:

```python
from merlin.forest import ExtraForestClassifier, ExtraForestRegressor, IsolationForest
from merlin.mondrian import MondrianForestClassifier, MondrianForestRegressor
```

These are subclasses of `RandomForest` with fixed `task`/`split` defaults.

| `split`      | `task`         | Wrapper class              |
| ------------ | -------------- | -------------------------- |
| `"best"`     | `"classifier"` | `ExtraForestClassifier`    |
| `"best"`     | `"regressor"`  | `ExtraForestRegressor`     |
| `"random"`   | `"anomaly"`    | `IsolationForest`          |
| `"mondrian"` | `"classifier"` | `MondrianForestClassifier` |
| `"mondrian"` | `"regressor"`  | `MondrianForestRegressor`  |

## Limitations vs official implementations

### ExtraForest vs scikit-learn ExtraTrees

merlin's `ExtraForestClassifier`/`ExtraForestRegressor` share the same core idea
— random split thresholds at each node — but differ in several ways:

- **No bootstrap by default**: ExtraTrees in sklearn defaults to
  `bootstrap=False`; merlin defaults to `bootstrap=True`. Set `bootstrap=False`
  to match sklearn.
- **Split evaluation**: sklearn samples `max_features` random candidates and
  picks the best split; merlin does the same but uses simpler purity/variance
  calculations. Numerical agreement is not expected.
- **Performance**: sklearn uses Cython-optimized tree routines. merlin converts
  data to Python lists internally. Expect 10–100× slower training on non-trivial
  data.
- **No `warm_start`**, no `ccp_alpha` (cost-complexity pruning), no
  `min_samples_leaf`/`min_samples_split` parameters, no `class_weight` support.
- **No OOB score**, no feature importance.

### IsolationForest vs scikit-learn IsolationForest

- **Same algorithm** (random split trees, path-length scoring). Numerical
  agreement is not expected due to different RNG.
- **No `warm_start`**, no support for `max_features` (always uses all features).
- **`contamination`** only supports `'auto'` and float values; the offset is set
  to `-0.5` for `'auto'`.
- Slower than sklearn's optimized Cython implementation.

### MondrianForest vs the original Mondrian Forest (Lakshminarayanan et al. 2014)

- **Simplified Mondrian process**: merlin uses exponential time sampling but
  does not implement the full hierarchical generative process described in the
  paper.
- **No true Mondrian properties**: the trees are not guaranteed to have the
  theoretical properties of Mondrian processes (e.g., the tree distribution does
  not follow the Mondrian process exactly).
- **`partial_fit` per-sample updates**: merlin updates one sample at a time per
  tree rather than batching. This is correct but slower.
- **No `predict_proba` smoothing** for zero-count classes (returns equal
  probability).
- **Lifetime parameter** controls tree depth but node creation in `partial_fit`
  follows a simpler logic than the paper's cascade.
- No support for `n_jobs` (always single-threaded).

## SHAP values

merlin includes a built-in TreeSHAP implementation for feature attribution — no
external library needed.

```python
from merlin.forest import RandomForest

rf = RandomForest(n_estimators=10, task="regressor", split="best", random_state=42)
rf.fit(X, y)

# Standalone call
shap_values, bias = rf.shap_values(X)
# shap_values shape: (n_samples, n_features)
# bias: expected prediction (float)

# Or via the output_shap flag
preds, shap_values, bias = rf.predict(X, output_shap=True)
```

For classifiers, `shap_values` returns per-class attributions:

```python
clf = RandomForest(n_estimators=10, task="classifier", split="best", random_state=42)
clf.fit(X, y)

shap_values, bias = clf.shap_values(X)
# shap_values shape: (n_samples, n_features, n_classes)
# bias: expected probability per class (list of length n_classes)

probs, shap_values, bias = clf.predict_proba(X, output_shap=True)
```

The SHAP decomposition satisfies `Σ φ + bias = prediction` for every sample.

### How it works

merlin implements the polynomial-time TreeSHAP algorithm (Lundberg et al. 2017)
using a per-leaf subset enumeration approach. Each leaf categorises path
features into _toward_ (x follows the same direction) and _away_ (x goes
opposite), and computes contributions using Shapley kernel weights that account
for both path and non-path features. Repeated features on a path are collapsed
automatically.

Key implementation details:

- **Complexity**: `O(T · L · 2^a)` per sample, where `T` = trees, `L` = leaves,
  `a` ≤ max unique toward features on a single path. For typical shallow trees
  (depth ≤ 10) this is very fast.
- **Bias**: Computed once per tree (independent of sample) as the expected value
  over the training distribution.
- **Classifier**: Runs TreeSHAP independently per class using normalised
  per-class probabilities from each leaf.
- **No dependencies**: Pure Python stdlib + `lib.array`.

### Limitations

- No support for anomaly detection (`task="anomaly"`) — SHAP values are only
  meaningful for regressor and classifier tasks.
- For deep trees (depth > 15) with many features, the per-leaf 2^a enumeration
  may become slow. Trees with ≤ 150 leaves and ≤ 10 unique toward features per
  leaf complete in milliseconds.
- Classifier SHAP runs one TreeSHAP pass per class, adding linear overhead in
  the number of classes.

### General

- All computations are in pure Python — no numpy, no C extensions.
- Only float64 and int32 array dtypes are used internally.
- Trees are stored as linked Python objects; serialization with `pickle` works
  but models can be large in memory.
- No GPU support, no parallelism.
