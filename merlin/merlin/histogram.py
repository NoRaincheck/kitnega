"""Histogram-based gradient boosting for classification and regression.

LightGBM-style implementation using histogram binning for fast split finding.
Two binning strategies: uniform (equal-width) and quantile (equal-frequency
quantization, similar to deep-learning quantization at 4-bit / 8-bit granularity).
"""

import math
import random

from lib.array import array, empty, unique

from merlin._core import _Node, _to_flat_list, _to_list, _traverse


def _bin_uniform(x, n_bins):
    lo = min(x)
    hi = max(x)
    if hi - lo < 1e-15:
        return [lo, hi], [0] * len(x)
    width = (hi - lo) / n_bins
    edges = [lo + i * width for i in range(n_bins + 1)]
    indices = []
    for v in x:
        b = int((v - lo) / width)
        if b >= n_bins:
            b = n_bins - 1
        indices.append(b)
    return edges, indices


def _bin_quantile(x, n_bins):
    if not x:
        return [], []
    xs = sorted(x)
    n = len(xs)
    if n <= n_bins:
        edges = [xs[0]]
        for v in xs[1:]:
            if v != edges[-1]:
                edges.append(v)
        edges.append(xs[-1])
        actual_bins = len(edges) - 1
        indices = []
        for v in x:
            b = 0
            for j in range(actual_bins - 1, -1, -1):
                if v >= edges[j]:
                    b = j
                    break
            indices.append(b)
        return edges, indices

    edges = [xs[0]]
    step = (n - 1) / n_bins
    for i in range(1, n_bins):
        idx = round(step * i)
        if xs[idx] != edges[-1]:
            edges.append(xs[idx])
    edges.append(xs[-1])

    actual_bins = len(edges) - 1
    indices = []
    for v in x:
        b = 0
        for j in range(actual_bins - 1, -1, -1):
            if v >= edges[j]:
                b = j
                break
        indices.append(b)
    return edges, indices


def _find_best_split_histogram(X_binned, gradients, hessians, sample_indices, n_bins, min_samples_leaf):
    n_features = len(X_binned[0])
    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for feature in range(n_features):
        right_grad = [0.0] * n_bins
        right_hess = [0.0] * n_bins
        right_count = [0] * n_bins
        for idx in sample_indices:
            b = X_binned[idx][feature]
            right_grad[b] += gradients[idx]
            right_hess[b] += hessians[idx]
            right_count[b] += 1

        sum_left_grad = 0.0
        sum_right_grad = sum(right_grad)
        sum_left_hess = 0.0
        sum_right_hess = sum(right_hess)
        total_left = 0
        total_right = len(sample_indices)

        for b in range(n_bins - 1):
            total_left += right_count[b]
            total_right -= right_count[b]
            sum_left_grad += right_grad[b]
            sum_right_grad -= right_grad[b]
            sum_left_hess += right_hess[b]
            sum_right_hess -= right_hess[b]

            if total_left < min_samples_leaf or total_right < min_samples_leaf:
                continue
            if sum_left_hess < 1e-15 or sum_right_hess < 1e-15:
                continue

            gain = 0.5 * (
                sum_left_grad**2 / sum_left_hess
                + sum_right_grad**2 / sum_right_hess
                - (sum_left_grad + sum_right_grad) ** 2 / (sum_left_hess + sum_right_hess)
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = b

    return best_feature, best_threshold, best_gain


class _HistogramTree:
    def __init__(self, max_depth, min_samples_leaf, rng):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = rng
        self.root = None
        self.X_binned = None
        self.gradients = None
        self.hessians = None
        self.n_bins = 0

    def fit(self, X_binned, gradients, hessians, sample_indices, n_bins):
        self.X_binned = X_binned
        self.gradients = gradients
        self.hessians = hessians
        self.n_bins = n_bins
        self.root = _Node()
        self._build(self.root, sample_indices, 0)

    def _build(self, node, sample_indices, depth):
        node.size = len(sample_indices)
        node.depth = depth

        sum_g = sum(self.gradients[idx] for idx in sample_indices)
        sum_h = sum(self.hessians[idx] for idx in sample_indices)
        node.prediction = -sum_g / max(sum_h, 1e-15)

        if self.max_depth is not None and depth >= self.max_depth:
            return
        if len(sample_indices) < 2 * self.min_samples_leaf:
            return

        best_f, best_b, best_gain = _find_best_split_histogram(
            self.X_binned,
            self.gradients,
            self.hessians,
            sample_indices,
            self.n_bins,
            self.min_samples_leaf,
        )
        if best_f is None or best_gain <= 0:
            return

        node.is_leaf = False
        node.feature = best_f
        node.threshold = best_b

        left_idx = [i for i in sample_indices if self.X_binned[i][best_f] <= best_b]
        right_idx = [i for i in sample_indices if self.X_binned[i][best_f] > best_b]

        node.left = _Node()
        node.right = _Node()
        self._build(node.left, left_idx, depth + 1)
        self._build(node.right, right_idx, depth + 1)

    def predict_leaf(self, X_binned):
        results = []
        for x in X_binned:
            node = _traverse(self.root, x)
            results.append(node.prediction)
        return results


class HistogramGradientBoosting:
    """Histogram-based gradient boosting classifier/regressor.

    Uses LightGBM-style histogram binning for fast split finding.

    Parameters
    ----------
    task : str
        ``"classifier"`` or ``"regressor"``.
    n_estimators : int
        Number of boosting iterations.
    learning_rate : float
        Shrinkage applied to each tree's contribution.
    max_depth : int or None
        Maximum tree depth. None defaults to 6.
    min_samples_leaf : int
        Minimum samples in a leaf node.
    n_bins : int
        Number of histogram bins.
    bin_strategy : str
        ``"uniform"`` (equal-width) or ``"quantile"`` (equal-frequency).
    subsample : float
        Fraction of rows used per tree (stochastic gradient boosting).
    random_state : int or None
        RNG seed.
    """

    def __init__(
        self,
        task="classifier",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=None,
        min_samples_leaf=20,
        n_bins=256,
        bin_strategy="uniform",
        subsample=1.0,
        random_state=None,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth if max_depth is not None else 6
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.subsample = subsample
        self.random_state = random_state
        self.trees_ = []
        self.bin_edges_ = []
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_in_ = 0
        self.initial_prediction_ = None

    def fit(self, X, y):
        X_list = _to_list(X)
        y_data = _to_flat_list(y)
        n = len(X_list)
        n_features = len(X_list[0]) if X_list else 0
        self.n_features_in_ = n_features

        bin_edges = []
        X_binned_cols = []
        for j in range(n_features):
            col = [X_list[i][j] for i in range(n)]
            if self.bin_strategy == "quantile":
                edges, indices = _bin_quantile(col, self.n_bins)
            else:
                edges, indices = _bin_uniform(col, self.n_bins)
            bin_edges.append(edges)
            X_binned_cols.append(indices)
        self.bin_edges_ = bin_edges

        X_binned = []
        for i in range(n):
            row = [X_binned_cols[j][i] for j in range(n_features)]
            X_binned.append(row)

        rng = random.Random(self.random_state)
        self.trees_ = []

        if self.task == "classifier":
            self.classes_ = unique(array(y_data))
            self.n_classes_ = len(self.classes_)
            class_map = {float(c): i for i, c in enumerate(self.classes_.flat)}
            y_idx = [class_map[float(v)] for v in y_data]

            counts = [0] * self.n_classes_
            for v in y_idx:
                counts[v] += 1
            self.initial_prediction_ = [math.log(max(c / n, 1e-15)) for c in counts]

            raw_scores = [list(self.initial_prediction_) for _ in range(n)]

            self.trees_ = [[] for _ in range(self.n_classes_)]

            for _ in range(self.n_estimators):
                gradients_k, hessians_k = self._compute_gradients_classifier(raw_scores, y_idx, n)

                if self.subsample < 1.0:
                    m = max(1, int(n * self.subsample))
                    sample_idx = rng.sample(range(n), m)
                else:
                    sample_idx = list(range(n))

                for k in range(self.n_classes_):
                    tree = _HistogramTree(
                        self.max_depth,
                        self.min_samples_leaf,
                        rng,
                    )
                    tree.fit(X_binned, gradients_k[k], hessians_k[k], sample_idx, self.n_bins)
                    self.trees_[k].append(tree)

                    leaf_vals = tree.predict_leaf(X_binned)
                    for i in range(n):
                        raw_scores[i][k] += self.learning_rate * leaf_vals[i]
        else:
            self.classes_ = None
            self.n_classes_ = 0
            self.initial_prediction_ = sum(y_data) / n if n else 0.0

            raw_scores = [self.initial_prediction_] * n
            self.trees_ = []

            for _ in range(self.n_estimators):
                gradients, hessians = self._compute_gradients_regressor(raw_scores, y_data, n)

                if self.subsample < 1.0:
                    m = max(1, int(n * self.subsample))
                    sample_idx = rng.sample(range(n), m)
                else:
                    sample_idx = list(range(n))

                tree = _HistogramTree(
                    self.max_depth,
                    self.min_samples_leaf,
                    rng,
                )
                tree.fit(X_binned, gradients, hessians, sample_idx, self.n_bins)
                self.trees_.append(tree)

                leaf_vals = tree.predict_leaf(X_binned)
                for i in range(n):
                    raw_scores[i] += self.learning_rate * leaf_vals[i]

        return self

    def _compute_gradients_classifier(self, raw_scores, y_idx, n):
        gradients_k = [[0.0] * n for _ in range(self.n_classes_)]
        hessians_k = [[0.0] * n for _ in range(self.n_classes_)]
        for i in range(n):
            scores = raw_scores[i]
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            total = sum(exps)
            probs = [e / total for e in exps]
            for k in range(self.n_classes_):
                target = 1.0 if k == y_idx[i] else 0.0
                gradients_k[k][i] = probs[k] - target
                hessians_k[k][i] = probs[k] * (1.0 - probs[k])
                if hessians_k[k][i] < 1e-15:
                    hessians_k[k][i] = 1e-15
        return gradients_k, hessians_k

    def _compute_gradients_regressor(self, predictions, y_data, n):
        gradients = [predictions[i] - y_data[i] for i in range(n)]
        hessians = [1.0] * n
        return gradients, hessians

    def _transform_bins(self, X):
        X_list = _to_list(X)
        n_features = len(X_list[0]) if X_list else 0
        X_binned = []
        for row in X_list:
            binned_row = []
            for j in range(min(n_features, len(self.bin_edges_))):
                edges = self.bin_edges_[j]
                actual_bins = len(edges) - 1
                v = row[j]
                b = 0
                for jj in range(actual_bins - 1, -1, -1):
                    if v >= edges[jj]:
                        b = jj
                        break
                binned_row.append(b)
            X_binned.append(binned_row)
        return X_binned

    def predict_proba(self, X):
        if self.task != "classifier":
            raise ValueError("predict_proba is only for classifiers")
        X_binned = self._transform_bins(X)
        n = len(X_binned)
        raw = [list(self.initial_prediction_) for _ in range(n)]
        for k in range(self.n_classes_):
            for tree in self.trees_[k]:
                leaf_vals = tree.predict_leaf(X_binned)
                for i in range(n):
                    raw[i][k] += self.learning_rate * leaf_vals[i]

        out = empty((n, self.n_classes_), "float64")
        for i in range(n):
            max_s = max(raw[i])
            exps = [math.exp(s - max_s) for s in raw[i]]
            total = sum(exps)
            for c in range(self.n_classes_):
                out[i, c] = exps[c] / total
        return out

    def predict(self, X):
        if self.task == "classifier":
            proba = self.predict_proba(X)
            out = empty((proba.shape[0],), "float64")
            for i in range(proba.shape[0]):
                best_c = 0
                for c in range(1, self.n_classes_):
                    if proba[i, c] > proba[i, best_c]:
                        best_c = c
                out[i] = self.classes_[best_c]
            return out
        else:
            X_binned = self._transform_bins(X)
            n = len(X_binned)
            raw = [self.initial_prediction_] * n
            for tree in self.trees_:
                leaf_vals = tree.predict_leaf(X_binned)
                for i in range(n):
                    raw[i] += self.learning_rate * leaf_vals[i]
            out = empty((n,), "float64")
            for i in range(n):
                out[i] = raw[i]
            return out
