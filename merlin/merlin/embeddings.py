"""Numerical feature embeddings for tabular data.

Two strategies:

1. Piecewise-linear (uniform) — divides each numeric feature into bins using
   uniform quantiles, then represents each value as a linear interpolation
   between the two adjacent bin boundaries.  Produces ``n_bins + 1`` output
   columns per input feature.

2. Tree-split boundaries — trains random-forest estimators on the data,
   collects all split points across trees for each feature, and uses those
   splits as non-uniform bin boundaries for piecewise-linear encoding.  This
   is inspired by the approach described in *"On Embeddings for Numerical
   Features in Tabular Deep Learning"* (Yandex Research, NeurIPS 2022).

Both strategies are implemented as fit/transform stateful objects and also
expose convenience functions that build and transform in one call.
"""

from typing import List, Optional

from lib.array import array, empty

from merlin._core import _to_list
from merlin.forest import ExtraForestRegressor

# ---------------------------------------------------------------------------
# Piecewise-linear (uniform quantile) encoding
# ---------------------------------------------------------------------------


def _quantile_boundaries(x: list[float], n_bins: int) -> list[float]:
    """Return ``n_bins + 1`` quantile-based boundaries for a single feature."""
    if not x:
        return []
    xs = sorted(x)
    n = len(xs)
    boundaries = [xs[0]]
    step = (n - 1) / n_bins
    for i in range(1, n_bins):
        idx = round(step * i)
        if xs[idx] != boundaries[-1]:
            boundaries.append(xs[idx])
    boundaries.append(xs[-1])
    return boundaries


def _piecewise_linear_encode(
    value: float, boundaries: list[float], out: list[float], offset: int
) -> None:
    """Encode a single scalar into two adjacent bins via linear interpolation.

    If ``value`` falls between ``boundaries[i]`` and ``boundaries[i+1]`` the
    output at ``offset + i`` gets ``(boundaries[i+1] - value) / width`` and
    ``offset + i + 1`` gets ``(value - boundaries[i]) / width``.

    Values outside the [min, max] range are clamped to the edge bin.
    """
    n = len(boundaries) - 1
    if n <= 0:
        return

    # Find which bracket ``value`` falls into
    for _i in range(n):
        if value >= boundaries[_i]:
            pass
        else:
            break

    # Find the first non-degenerate bracket containing value
    for j in range(n):
        if boundaries[j] != boundaries[j + 1]:
            lo_j, hi_j = boundaries[j], boundaries[j + 1]
            if lo_j <= value <= hi_j:
                width = hi_j - lo_j
                out[offset + j] = (hi_j - value) / width
                out[offset + j + 1] = (value - lo_j) / width
                return
    # Value is outside all brackets — clamp to nearest edge bracket
    first_valid = None
    last_valid = None
    for j in range(n):
        if boundaries[j] != boundaries[j + 1]:
            if first_valid is None:
                first_valid = j
            last_valid = j
    if first_valid is not None:
        lo_v = boundaries[first_valid]
        if value < lo_v:
            out[offset + first_valid] = 1.0
        else:
            out[offset + last_valid + 1] = 1.0
    # else all brackets are degenerate — put weight on first column
    elif n > 0:
        out[offset] = 1.0


class PiecewiseLinearEncoder:
    """Piecewise-linear encoding using uniform quantile bins.

    Parameters
    ----------
    n_bins : int
        Number of bins per feature (produces ``n_bins + 1`` output columns).
    random_state : int or None

    Attributes
    ----------
    boundaries_ : list[list[float]]
        Per-feature sorted boundary lists fitted during ``.fit()``.
    """

    def __init__(self, n_bins: int = 8, random_state=None):
        self.n_bins = n_bins
        self.random_state = random_state
        self.boundaries_: List[List[float]] = []

    # -- fit ---------------------------------------------------------------

    def fit(self, X) -> "PiecewiseLinearEncoder":
        """Compute quantile boundaries from training data."""
        X_list = _to_list(X)
        n_features = len(X_list[0]) if X_list else 0
        self.boundaries_ = []
        for j in range(n_features):
            col = [row[j] for row in X_list]
            self.boundaries_.append(_quantile_boundaries(col, self.n_bins))
        return self

    def fit_transform(self, X) -> "PiecewiseLinearEncoder":
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    # -- transform ---------------------------------------------------------

    def transform(self, X) -> array:
        """Transform data using fitted boundaries.

        Returns an ``ndarray`` of shape ``(n_samples, n_features * (n_bins + 1))``.
        """
        X_list = _to_list(X)
        n_samples = len(X_list)
        if not self.boundaries_:
            raise RuntimeError("Must call fit() before transform()")

        out_dim = sum(len(b) - 1 for b in self.boundaries_) + len(self.boundaries_)
        embedding = empty((n_samples, out_dim), "float64")

        col_offset = 0
        for j, boundaries in enumerate(self.boundaries_):
            n_bins_j = len(boundaries) - 1
            if n_bins_j <= 0:
                continue
            stride = n_bins_j + 1
            for i, x in enumerate(X_list):
                _piecewise_linear_encode(x[j], boundaries, embedding[i], col_offset)
            col_offset += stride

        return embedding


def piecewise_linear_embedding(
    X, y=None, n_bins: int = 8, random_state=None
) -> array:
    """Convenience function — fit and transform in one call."""
    enc = PiecewiseLinearEncoder(n_bins=n_bins, random_state=random_state)
    return enc.fit_transform(X)


# ---------------------------------------------------------------------------
# Piecewise-linear (tree-split boundaries) encoding
# ---------------------------------------------------------------------------


def _collect_tree_splits(
    forest_trees: list, n_features: int, rng_seed: Optional[int] = None
) -> List[List[float]]:
    """Collect all split thresholds from a fitted forest for each feature."""
    splits_per_feature: List[List[float]] = [[] for _ in range(n_features)]

    for tree in forest_trees:
        def _walk(node):
            if node is None or node.is_leaf:
                return
            if node.feature is not None and node.threshold is not None:
                splits_per_feature[node.feature].append(node.threshold)
            _walk(node.left)
            _walk(node.right)

        _walk(tree.root)

    # Deduplicate and sort per feature
    result = []
    for s in splits_per_feature:
        unique_s = sorted(set(round(v, 10) for v in s))
        result.append(unique_s)
    return result


def _forest_boundaries(
    splits: list[float], n_bins: int, fallback_min=None, fallback_max=None
) -> list[float]:
    """Convert raw split points into a dense set of bin boundaries.

    Uses the sorted unique tree-split thresholds as boundaries; if there are
    are fewer than ``n_bins + 1`` boundaries, add uniform quantile-like
    midpoints between adjacent splits to reach at least ``n_bins + 1`` total
    boundaries.

    If no splits were found (empty list), returns a single-element list as a
    fallback — the encoder will produce one column with value 1.0 for all
    samples on this feature.
    """
    if not splits:
        return []

    # Start with the raw sorted unique splits
    boundaries = sorted(set(round(v, 10) for v in splits))

    # If we don't have enough bins, insert midpoints between adjacent non-degenerate splits
    while len(boundaries) - 1 < n_bins:
        new_bounds = []
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo > 1e-15:
                mid = (lo + hi) / 2.0
                new_bounds.append(mid)
        if not new_bounds:
            break
        # Merge and deduplicate maintaining sort order
        merged = set(boundaries)
        for b in new_bounds:
            merged.add(round(b, 10))
        boundaries = sorted(merged)

    return boundaries


class PiecewiseLinearForestEncoder:
    """Piecewise-linear encoding using forest-derived split boundaries.

    Trains random-forest regressors on the data, collects all split thresholds
    from each feature across trees, and uses those splits as bin boundaries for
    piecewise-linear encoding. Inspired by the approach described in *
    "On Embeddings for Numerical Features in Tabular Deep Learning"* (Yandex
    Research, NeurIPS 2022).

    Parameters
    ----------
    n_estimators : int
        Number of trees in the auxiliary forest (default 20).
    max_depth : int or None
        Max depth per tree. ``None`` means grow until leaves are pure
        (or singletons).
    n_bins : int
        Maximum number of bins per feature for the piecewise-linear encoding.
        The actual number may be less if fewer splits exist.  Default 8.
    max_samples : float or None
        Fraction of rows to use when fitting each tree (subsample with
        replacement).  ``None`` means full data.
    random_state : int or None

    Attributes
    ----------
    boundaries_ : list[list[float]]
        Per-feature bin boundaries fitted during ``.fit()``.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: Optional[int] = None,
        n_bins: int = 8,
        max_samples: Optional[float] = None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.max_samples = max_samples
        self.random_state = random_state
        self.boundaries_: List[List[float]] = []

    # -- fit ---------------------------------------------------------------

    def fit(self, X) -> "PiecewiseLinearForestEncoder":
        """Train auxiliary forest and compute split-based boundaries."""
        X_list = _to_list(X)
        n_samples = len(X_list)
        n_features = len(X_list[0]) if X_list else 0

        # Build a dummy y for the auxiliary regressor (all zeros — we only
        # care about tree structure, not predictions).
        y_dummy = [0.0] * n_samples

        forest = ExtraForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            bootstrap=True,
            random_state=self.random_state,
        )
        forest.fit(X_list, y_dummy)

        # Collect splits from all trees
        raw_splits = _collect_tree_splits(forest.trees_, n_features)

        # Compute per-feature min/max for fallback boundaries on features with no splits
        feat_min = [min(row[j] for row in X_list) for j in range(n_features)]
        feat_max = [max(row[j] for row in X_list) for j in range(n_features)]

        # Convert to dense boundaries per feature
        self.boundaries_ = []
        for j in range(n_features):
            if raw_splits[j]:
                self.boundaries_.append(
                    _forest_boundaries(raw_splits[j], self.n_bins, feat_min[j], feat_max[j])
                )
            else:
                # No splits found — use min/max as a single bin boundary pair
                self.boundaries_.append([feat_min[j], feat_max[j]])

        return self

    def fit_transform(self, X) -> "PiecewiseLinearForestEncoder":
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    # -- transform ---------------------------------------------------------

    def transform(self, X) -> array:
        """Transform data using fitted tree-based boundaries.

        Returns an ``ndarray`` of shape ``(n_samples, sum(n_bins_j + 1))``.
        """
        if not self.boundaries_:
            raise RuntimeError("Must call fit() before transform()")

        X_list = _to_list(X)
        n_samples = len(X_list)
        out_dim = sum(len(b) - 1 for b in self.boundaries_) + len(self.boundaries_)
        embedding = empty((n_samples, out_dim), "float64")

        col_offset = 0
        for j, boundaries in enumerate(self.boundaries_):
            n_bins_j = len(boundaries) - 1
            if n_bins_j <= 0:
                continue
            stride = n_bins_j + 1
            for i, x in enumerate(X_list):
                _piecewise_linear_encode(x[j], boundaries, embedding[i], col_offset)
            col_offset += stride

        return embedding


def piecewise_linear_forest_embedding(
    X, y=None, n_estimators: int = 20, max_depth=None, n_bins: int = 8, random_state=None
) -> array:
    """Convenience function — fit and transform in one call."""
    enc = PiecewiseLinearForestEncoder(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_bins=n_bins,
        random_state=random_state,
    )
    return enc.fit_transform(X)


# ---------------------------------------------------------------------------
# Unified embedding factory (switch between strategies via parameter)
# ---------------------------------------------------------------------------

_EncodingStrategy = str  # 'piecewise-linear' | 'tree-split'


def numerical_embedding(
    X,
    y=None,
    strategy: _EncodingStrategy = "piecewise-linear",
    n_bins: int = 8,
    random_state=None,
    **kwargs,
) -> array:
    """Create a numerical embedding with the given strategy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numerical feature matrix.
    y : ignored
        Present for API compatibility.
    strategy : str
        ``'piecewise-linear'`` — uniform quantile bins.
        ``'tree-split'`` — forest-derived split boundaries.
    n_bins : int
        Target number of bins per feature (may differ slightly in practice).
    random_state : int or None
        Seed for reproducibility.
    **kwargs
        Passed to the underlying encoder constructor:

        * For ``'piecewise-linear'`` — no extra kwargs.
        * For ``'tree-split'`` — ``n_estimators``, ``max_depth``,
          ``max_samples`` (fraction).

    Returns
    -------
    embedding : ndarray of shape ``(n_samples, total_output_dim)``
    """
    if strategy == "piecewise-linear":
        enc = PiecewiseLinearEncoder(n_bins=n_bins, random_state=random_state)
    elif strategy == "tree-split":
        n_estimators = kwargs.get("n_estimators", 20)
        max_depth = kwargs.get("max_depth")
        max_samples = kwargs.get("max_samples")
        enc = PiecewiseLinearForestEncoder(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_bins=n_bins,
            max_samples=max_samples,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            "Use 'piecewise-linear' or 'tree-split'."
        )

    return enc.fit_transform(X)


# ---------------------------------------------------------------------------
# Embedding dimension helpers
# ---------------------------------------------------------------------------


def piecewise_linear_output_dim(n_features: int, n_bins: int) -> int:
    """Compute output dimension for uniform quantile encoding."""
    return n_features * (n_bins + 1)


def estimate_piecewise_linear_forest_output_dim(
    X, n_estimators: int = 20, max_depth=None, n_bins: int = 8, random_state=None
) -> int:
    """Estimate output dimension for forest-based piecewise-linear encoding without full transform."""
    X_list = _to_list(X)
    n_features = len(X_list[0]) if X_list else 0

    y_dummy = [0.0] * len(X_list)
    forest = ExtraForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        bootstrap=True,
        random_state=random_state,
    )
    forest.fit(X_list, y_dummy)

    raw_splits = _collect_tree_splits(forest.trees_, n_features)
    total_dim = 0
    for j in range(n_features):
        b = _forest_boundaries(raw_splits[j], n_bins)
        total_dim += max(1, len(b) - 1 + 1)  # at least one output column per feature

    return total_dim
