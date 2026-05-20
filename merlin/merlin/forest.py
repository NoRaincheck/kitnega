import copy
import math
import random

from lib.array import array, empty, unique

from merlin._core import _Node, _to_flat_list, _to_list, _traverse
from merlin._shap import forest_shap_values

_EULER_GAMMA = 0.5772156649015329


def _average_path_length(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + _EULER_GAMMA) - 2.0 * (n - 1) / n


def _resolve_max_features(max_features, n_features):
    if max_features is None:
        return n_features
    if isinstance(max_features, int):
        return min(max_features, n_features)
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    if max_features == "sqrt":
        return max(1, int(math.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(math.log2(n_features)))
    return n_features


def _gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    g = 1.0
    for c in counts.values():
        p = c / n
        g -= p * p
    return g


def _gini_gain(y, ly, ry):
    total = len(y)
    return _gini(y) - (len(ly) / total) * _gini(ly) - (len(ry) / total) * _gini(ry)


def _var(y):
    n = len(y)
    if n <= 1:
        return 0.0
    m = sum(y) / n
    return sum((v - m) ** 2 for v in y) / n


def _var_gain(y, ly, ry):
    total = len(y)
    return _var(y) - (len(ly) / total) * _var(ly) - (len(ry) / total) * _var(ry)


def _is_homogeneous(X):
    if len(X) <= 1:
        return True
    first = tuple(X[0])
    for r in X[1:]:
        if tuple(r) != first:
            return False
    return True


class _Tree:
    def __init__(self, split="best", task="classifier", max_depth=None,
                 max_features=None, lifetime=float("inf"), random_state=None):
        self.split = split
        self.task = task
        self.max_depth = max_depth
        self.max_features = max_features
        self.lifetime = lifetime
        self.rng = random.Random(random_state)
        self.root = None
        self.n_classes = None

    def fit(self, X, y=None, n_classes=None):
        X_list = _to_list(X)
        if self.task == "classifier":
            y_list = _to_flat_list(y)
            self.n_classes = n_classes
        elif self.task == "regressor":
            y_list = _to_flat_list(y)
        else:
            y_list = None

        if self.split == "mondrian":
            self.root = _Node(tau=0.0)
            self._build_mondrian(self.root, X_list, y_list, 0)
        else:
            self.root = self._build(X_list, y_list, 0)
        return self

    def _build(self, X, y, depth):
        node = _Node()
        n = len(X)
        node.size = n
        node.depth = depth

        if self.task == "classifier":
            counts = [0] * self.n_classes
            for v in y:
                counts[int(v)] += 1
            node.prediction = counts
        elif self.task == "regressor":
            node.prediction = sum(y) / n if n else 0.0
        else:
            node.prediction = 0.0

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n <= 1:
            return node
        if _is_homogeneous(X):
            return node

        nf = len(X[0])

        if self.split == "random":
            lo = [float("inf")] * nf
            hi = [float("-inf")] * nf
            for x in X:
                for j in range(nf):
                    v = x[j]
                    if v < lo[j]:
                        lo[j] = v
                    if v > hi[j]:
                        hi[j] = v

            candidates = [(j, lo[j], hi[j]) for j in range(nf) if hi[j] - lo[j] > 1e-15]
            if not candidates:
                return node

            feature, lo_v, hi_v = self.rng.choice(candidates)
            threshold = self.rng.uniform(lo_v, hi_v)

            lx, ly, rx, ry = [], [], [], []
            for i, x in enumerate(X):
                if x[feature] <= threshold:
                    lx.append(x)
                    if y is not None:
                        ly.append(y[i])
                else:
                    rx.append(x)
                    if y is not None:
                        ry.append(y[i])

            if not lx or not rx:
                return node

        elif self.split == "best":
            n_try = _resolve_max_features(self.max_features, nf)
            features = list(range(nf))
            if n_try < nf:
                features = self.rng.sample(features, n_try)

            best_gain = -1.0
            best_feature = None
            best_threshold = None
            best_lx = best_ly = best_rx = best_ry = None

            for f in features:
                vals = [x[f] for x in X]
                lo, hi = min(vals), max(vals)
                if hi - lo < 1e-15:
                    continue
                threshold = self.rng.uniform(lo, hi)

                lx, ly, rx, ry = [], [], [], []
                for i, x in enumerate(X):
                    if x[f] <= threshold:
                        lx.append(x)
                        if y is not None:
                            ly.append(y[i])
                    else:
                        rx.append(x)
                        if y is not None:
                            ry.append(y[i])

                if not lx or not rx:
                    continue

                if self.task == "classifier":
                    gain = _gini_gain(y, ly, ry)
                else:
                    gain = _var_gain(y, ly, ry)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = threshold
                    best_lx, best_ly = lx, ly
                    best_rx, best_ry = rx, ry

            if best_feature is None:
                return node

            feature, threshold = best_feature, best_threshold
            lx, ly, rx, ry = best_lx, best_ly, best_rx, best_ry

        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(lx, ly if y is not None else None, depth + 1)
        node.right = self._build(rx, ry if y is not None else None, depth + 1)
        return node

    def _build_mondrian(self, node, X, y, depth):
        node.depth = depth
        node.size = len(X)
        node.update_bounds(X)

        if self.task == "classifier":
            counts = [0] * self.n_classes
            for v in y:
                counts[int(v)] += 1
            node.prediction = counts
        elif self.task == "regressor":
            node.prediction = sum(y) / len(y) if len(y) > 0 else 0.0

        nf = len(X[0])
        ranges = [node.upper_bounds[j] - node.lower_bounds[j] for j in range(nf)]
        total_range = sum(ranges)

        if total_range <= 1e-15 or len(X) <= 1:
            return

        rate = total_range
        delta = -math.log(self.rng.random()) / rate

        if node.tau + delta >= self.lifetime:
            return

        probs = [r / total_range for r in ranges]
        u = self.rng.random()
        cum = 0.0
        feature = 0
        for j, p in enumerate(probs):
            cum += p
            if u <= cum:
                feature = j
                break

        split_min = node.lower_bounds[feature]
        split_max = node.upper_bounds[feature]
        threshold = self.rng.uniform(split_min, split_max)

        lx, ly, rx, ry = [], [], [], []
        for i, x in enumerate(X):
            if x[feature] <= threshold:
                lx.append(x)
                if y is not None:
                    ly.append(y[i])
            else:
                rx.append(x)
                if y is not None:
                    ry.append(y[i])

        if not lx or not rx:
            return

        node.feature = feature
        node.threshold = threshold
        node.delta = delta
        node.is_leaf = False

        node.left = _Node(tau=node.tau + delta)
        node.right = _Node(tau=node.tau + delta)
        self._build_mondrian(node.left, lx, ly, depth + 1)
        self._build_mondrian(node.right, rx, ry, depth + 1)

    def _path_length(self, x):
        node = self.root
        depth = 0
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
            depth += 1
        return depth + _average_path_length(node.size)

    def predict(self, X):
        X_list = _to_list(X)
        if self.task == "classifier":
            out = empty((len(X_list), self.n_classes), "float64")
            for i, x in enumerate(X_list):
                node = _traverse(self.root, x)
                if node.prediction is None or sum(node.prediction) == 0:
                    for j in range(self.n_classes):
                        out[i, j] = 1.0 / self.n_classes
                else:
                    total = sum(node.prediction)
                    for j in range(self.n_classes):
                        out[i, j] = node.prediction[j] / total
            return out
        elif self.task == "regressor":
            out = empty((len(X_list),), "float64")
            for i, x in enumerate(X_list):
                node = _traverse(self.root, x)
                out[i] = node.prediction
            return out
        else:
            out = empty((len(X_list),), "float64")
            for i, x in enumerate(X_list):
                out[i] = self._path_length(x)
            return out

    def partial_fit(self, X, y):
        if self.split != "mondrian":
            raise NotImplementedError("partial_fit only supports mondrian trees")
        X_list = _to_list(X)
        y_list = _to_flat_list(y)
        if self.root is None:
            self.fit(X_list, y_list, n_classes=self.n_classes)
            return
        for i in range(len(X_list)):
            self._update(self.root, [X_list[i]], [y_list[i]])

    def _update(self, node, X, y):
        node.size += 1
        x = X[0]
        y0 = y[0]

        if self.task == "classifier":
            if node.prediction is None:
                node.prediction = [0] * self.n_classes
            node.prediction[int(y0)] += 1
        elif self.task == "regressor":
            if node.prediction is None:
                node.prediction = y0
            else:
                n = node.size
                node.prediction = ((n - 1) * node.prediction + y0) / n

        nf = len(x)
        if node.lower_bounds is None or node.upper_bounds is None:
            node.update_bounds(X)
            old_lower = list(node.lower_bounds)
            old_upper = list(node.upper_bounds)
        else:
            old_lower = list(node.lower_bounds)
            old_upper = list(node.upper_bounds)
            node.update_bounds(X)

        extension = [
            max(old_lower[j] - node.lower_bounds[j], 0) +
            max(node.upper_bounds[j] - old_upper[j], 0)
            for j in range(nf)
        ]
        total_extension = sum(extension)

        if node.is_leaf:
            if total_extension > 1e-15:
                delta = -math.log(self.rng.random()) / total_extension
                if node.tau + delta < self.lifetime:
                    probs = [e / total_extension for e in extension]
                    u = self.rng.random()
                    cum = 0.0
                    feature = 0
                    for j, p in enumerate(probs):
                        cum += p
                        if u <= cum:
                            feature = j
                            break

                    x_val = x[feature]
                    if x_val < old_lower[feature]:
                        threshold = self.rng.uniform(x_val, old_lower[feature])
                        new_node = _Node(tau=node.tau + delta)
                        new_node.left = _Node(tau=node.tau + delta)
                        new_node.right = copy.deepcopy(node)
                        new_node.right.tau = node.tau + delta
                    else:
                        threshold = self.rng.uniform(old_upper[feature], x_val)
                        new_node = _Node(tau=node.tau + delta)
                        new_node.left = copy.deepcopy(node)
                        new_node.left.tau = node.tau + delta
                        new_node.right = _Node(tau=node.tau + delta)

                    new_node.feature = feature
                    new_node.threshold = threshold
                    new_node.delta = delta
                    new_node.is_leaf = False
                    new_node.lower_bounds = node.lower_bounds
                    new_node.upper_bounds = node.upper_bounds
                    node.feature = new_node.feature
                    node.threshold = new_node.threshold
                    node.left = new_node.left
                    node.right = new_node.right
                    node.is_leaf = new_node.is_leaf
                    node.delta = new_node.delta
                    node.lower_bounds = new_node.lower_bounds
                    node.upper_bounds = new_node.upper_bounds
                    node.tau = new_node.tau
        else:
            if x[node.feature] <= node.threshold:
                self._update(node.left, X, y)
            else:
                self._update(node.right, X, y)


class RandomForest:
    def __init__(self, n_estimators=100, task="classifier", split="best",
                 max_depth=None, max_features=None, bootstrap=True,
                 lifetime=float("inf"), max_samples=None,
                 contamination="auto", random_state=None):
        self.n_estimators = n_estimators
        self.task = task
        self.split = split
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.lifetime = lifetime
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None
        self.n_classes_ = 0
        self.n_features_in_ = 0
        self.offset_ = -0.5

    def fit(self, X, y=None):
        X_list = _to_list(X)
        self.n_features_in_ = len(X_list[0]) if X_list else 0

        if self.task == "classifier":
            self.classes_ = unique(y)
            self.n_classes_ = len(self.classes_)
            class_to_idx = {float(c): i for i, c in enumerate(self.classes_.flat)}
            y_flat = _to_flat_list(y)
            y_data = [class_to_idx[float(v)] for v in y_flat]
            n_classes = self.n_classes_
        elif self.task == "regressor":
            y_data = _to_flat_list(y)
            n_classes = None
        else:
            y_data = None
            n_classes = None

        n = len(X_list)
        if self.task == "anomaly":
            if self.max_samples == "auto":
                sample_size = min(256, n)
            elif isinstance(self.max_samples, int):
                sample_size = min(self.max_samples, n)
            else:
                sample_size = n
            max_depth = self.max_depth if self.max_depth is not None else int(math.ceil(math.log2(sample_size)))
            self._c = _average_path_length(sample_size)
        else:
            max_depth = self.max_depth
            sample_size = n

        rng = random.Random(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = _Tree(
                split=self.split,
                task=self.task,
                max_depth=max_depth,
                max_features=self.max_features,
                lifetime=self.lifetime,
                random_state=rng.randint(0, 2**31 - 1),
            )

            if self.task == "anomaly":
                if sample_size < n:
                    subset = rng.sample(X_list, sample_size)
                    tree.fit(subset)
                else:
                    tree.fit(X_list)
            elif self.bootstrap:
                boot_X, boot_y = [], []
                for _ in range(n):
                    idx = rng.randint(0, n - 1)
                    boot_X.append(X_list[idx])
                    boot_y.append(y_data[idx])
                tree.fit(boot_X, boot_y, n_classes=n_classes)
            else:
                tree.fit(X_list, y_data, n_classes=n_classes)

            self.trees_.append(tree)
        return self

    def predict_proba(self, X, output_shap=False):
        if self.task != "classifier":
            raise ValueError("predict_proba is only for classifiers")
        acc = None
        for tree in self.trees_:
            p = tree.predict(X)
            if acc is None:
                acc = array([[0.0] * self.n_classes_ for _ in range(len(X))])
            for i in range(len(X)):
                for j in range(self.n_classes_):
                    acc[i, j] += p[i, j]
        n = len(self.trees_)
        for i in range(acc.shape[0]):
            for j in range(acc.shape[1]):
                acc[i, j] /= n
        if output_shap:
            shap, bias = forest_shap_values(self, X)
            return acc, shap, bias
        return acc

    def predict(self, X, output_shap=False):
        if self.task == "classifier":
            proba = self.predict_proba(X)
            out = empty((len(X),), "float64")
            for i in range(len(X)):
                best_j, best_v = 0, proba[i, 0]
                for j in range(1, self.n_classes_):
                    if proba[i, j] > best_v:
                        best_v = proba[i, j]
                        best_j = j
                out[i] = self.classes_[best_j]
            if output_shap:
                shap, bias = forest_shap_values(self, X)
                return out, shap, bias
            return out
        if self.task == "anomaly":
            scores = self.score_samples(X)
            out = empty((len(X),), "float64")
            for i in range(len(X)):
                out[i] = -1.0 if scores[i] < self.offset_ else 1.0
            return out
        n = len(X)
        sums = [0.0] * n
        for tree in self.trees_:
            p = tree.predict(X)
            for i in range(n):
                sums[i] += p[i]
        n_trees = len(self.trees_)
        out = empty((n,), "float64")
        for i in range(n):
            out[i] = sums[i] / n_trees
        if output_shap:
            shap, bias = forest_shap_values(self, X)
            return out, shap, bias
        return out

    def shap_values(self, X):
        """Compute SHAP values for all samples.

        Returns (shap_values, bias).

        Regressor:
            shap_values shape (n, n_features), bias is float.
        Classifier:
            shap_values shape (n, n_features, n_classes), bias is list
            of length n_classes.
        """
        return forest_shap_values(self, X)

    def score_samples(self, X):
        if self.task != "anomaly":
            raise ValueError("score_samples is only for anomaly detection")
        X_list = _to_list(X)
        out = empty((len(X_list),), "float64")
        for i, x in enumerate(X_list):
            depths = [tree.predict([x])[0] for tree in self.trees_]
            avg_depth = sum(depths) / len(depths)
            out[i] = 2.0 ** (-avg_depth / self._c) if self._c > 0 else 0.0
        return out

    def decision_function(self, X):
        if self.task != "anomaly":
            raise ValueError("decision_function is only for anomaly detection")
        return self.score_samples(X) - self.offset_

    def partial_fit(self, X, y, classes=None):
        if self.split != "mondrian":
            raise NotImplementedError("partial_fit only supports mondrian trees")
        X_list = _to_list(X)
        y_list = _to_flat_list(y)

        if self.classes_ is None:
            if classes is not None:
                self.classes_ = array(classes)
            else:
                self.classes_ = unique(array(y_list))
            self.n_classes_ = len(self.classes_)
            self.n_features_in_ = len(X_list[0])
            self.trees_ = []
            rng = random.Random(self.random_state)
            for _ in range(self.n_estimators):
                tree = _Tree(
                    split="mondrian",
                    task=self.task,
                    lifetime=self.lifetime,
                    random_state=rng.randint(0, 2**31 - 1),
                )
                tree.n_classes = self.n_classes_
                self.trees_.append(tree)

        class_to_idx = {float(c): i for i, c in enumerate(self.classes_.flat)}
        y_enc = [class_to_idx.get(float(v), 0) for v in y_list]
        for i in range(len(X_list)):
            for tree in self.trees_:
                tree.partial_fit([X_list[i]], [y_enc[i]])
        return self


class ExtraForestClassifier(RandomForest):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 bootstrap=True, random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            task="classifier",
            split="best",
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
        )


class ExtraForestRegressor(RandomForest):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 bootstrap=True, random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            task="regressor",
            split="best",
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
        )


class IsolationForest(RandomForest):
    def __init__(self, n_estimators=100, max_samples="auto",
                 contamination="auto", random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            task="anomaly",
            split="random",
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )


class MondrianForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, lifetime=float("inf"), random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            task="classifier",
            split="mondrian",
            lifetime=lifetime,
            random_state=random_state,
        )


class MondrianForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, lifetime=float("inf"), random_state=None):
        super().__init__(
            n_estimators=n_estimators,
            task="regressor",
            split="mondrian",
            lifetime=lifetime,
            random_state=random_state,
        )
