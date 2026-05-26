import math

from merlin._core import _copy_node
from merlin.forest import (
    ExtraForestClassifier,
    ExtraForestRegressor,
    IsolationForest,
    MondrianForestClassifier,
    MondrianForestRegressor,
    _Tree,
)

_EULER_GAMMA = 0.5772156649015329


def _is_forest(forest, task):
    if not hasattr(forest, "trees_") or not forest.trees_:
        raise ValueError("source forest must have fitted trees_")
    if forest.task != task:
        raise TypeError(f"expected {task} forest")


def extra_to_mondrian(extra_forest, lifetime=float("inf")):
    if not isinstance(extra_forest, (ExtraForestClassifier, ExtraForestRegressor)):
        raise TypeError("expected ExtraForestClassifier or ExtraForestRegressor")
    _is_forest(extra_forest, extra_forest.task)

    is_classifier = isinstance(extra_forest, ExtraForestClassifier)

    if is_classifier:
        mf = MondrianForestClassifier(
            n_estimators=len(extra_forest.trees_),
            lifetime=lifetime,
            random_state=extra_forest.random_state,
        )
        mf.classes_ = extra_forest.classes_
        mf.n_classes_ = extra_forest.n_classes_
        mf.n_features_in_ = extra_forest.n_features_in_
    else:
        mf = MondrianForestRegressor(
            n_estimators=len(extra_forest.trees_),
            lifetime=lifetime,
            random_state=extra_forest.random_state,
        )
        mf.n_features_in_ = extra_forest.n_features_in_

    mf.trees_ = []
    for et in extra_forest.trees_:
        mt = _Tree(split="mondrian", task=extra_forest.task, lifetime=lifetime)
        mt.n_classes = et.n_classes if is_classifier else None
        mt.root = _copy_node(et.root)
        mf.trees_.append(mt)
    return mf


def mondrian_to_extra(mondrian_forest):
    if not isinstance(mondrian_forest, (MondrianForestClassifier, MondrianForestRegressor)):
        raise TypeError("expected MondrianForestClassifier or MondrianForestRegressor")
    _is_forest(mondrian_forest, mondrian_forest.task)

    is_classifier = isinstance(mondrian_forest, MondrianForestClassifier)

    if is_classifier:
        ef = ExtraForestClassifier(
            n_estimators=len(mondrian_forest.trees_),
            bootstrap=False,
            random_state=mondrian_forest.random_state,
        )
        ef.classes_ = mondrian_forest.classes_
        ef.n_classes_ = mondrian_forest.n_classes_
        ef.n_features_in_ = mondrian_forest.n_features_in_
    else:
        ef = ExtraForestRegressor(
            n_estimators=len(mondrian_forest.trees_),
            bootstrap=False,
            random_state=mondrian_forest.random_state,
        )
        ef.n_features_in_ = mondrian_forest.n_features_in_

    ef.trees_ = []
    for mt in mondrian_forest.trees_:
        et = _Tree(split="best", task=mondrian_forest.task)
        et.n_classes = mt.n_classes if is_classifier else None
        et.root = _copy_node(mt.root)
        ef.trees_.append(et)
    return ef


def extra_to_iso(extra_forest, contamination="auto"):
    if not isinstance(extra_forest, (ExtraForestClassifier, ExtraForestRegressor)):
        raise TypeError("expected ExtraForestClassifier or ExtraForestRegressor")
    _is_forest(extra_forest, extra_forest.task)

    iso = IsolationForest(
        n_estimators=len(extra_forest.trees_),
        contamination=contamination,
        random_state=extra_forest.random_state,
    )

    iso.trees_ = []
    total = 0
    for et in extra_forest.trees_:
        it = _Tree(split="random", task="anomaly")
        it.root = _copy_node(et.root)
        iso.trees_.append(it)
        total += it.root.size

    sample_size = max(2, total // len(iso.trees_)) if iso.trees_ else 256
    c = 2.0 * (math.log(sample_size - 1) + _EULER_GAMMA) - 2.0 * (sample_size - 1) / sample_size
    iso._c = c
    iso.offset_ = -0.5
    return iso


def _count_leaves(node):
    """Count leaf nodes in a tree."""
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)


def _sample_to_leaf_index(root, x):
    """Traverse a single tree for one sample, return (leaf_idx, total_leaves)."""
    n_leaves = _count_leaves(root)
    if n_leaves <= 1:
        return 0, max(1, n_leaves)

    # Find which leaf index this sample reached by walking the tree
    def _find_leaf(node, x, idx):
        if node.is_leaf:
            return idx
        if x[node.feature] <= node.threshold:
            left_count = _count_leaves(node.left)
            return _find_leaf(node.left, x, idx)
        else:
            left_count = _count_leaves(node.left)
            return _find_leaf(node.right, x, idx + left_count)

    leaf_idx = _find_leaf(root, x, 0)
    return leaf_idx, n_leaves


def forest_to_embedding(forest, X):
    """Transform a fitted tree-based forest into a one-hot leaf indicator embedding.

    Each sample is transformed by traversing every tree in the forest. The result
    is a sparse binary encoding where each tree contributes ``max_leaves`` columns,
    with exactly one column set to 1.0 (the leaf reached by that sample).

    This follows sklearn's RandomTreesEmbedding semantics but operates on an
    already-trained model rather than building new trees.

    Parameters
    ----------
    forest : RandomForest subclass
        A fitted forest with ``trees_`` attribute containing _Tree instances.
    X : array-like of shape (n_samples, n_features)
        Data to transform.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_estimators * max_leaves_per_tree)
        One-hot encoded leaf indicators. Each row has exactly one 1.0 per tree.

    Examples
    --------
    >>> from merlin.forest import ExtraForestClassifier
    >>> from merlin.convert import forest_to_embedding
    >>> clf = ExtraForestClassifier(n_estimators=5, max_depth=3)
    >>> clf.fit(X, y)
    >>> emb = forest_to_embedding(clf, X_new)
    """
    if not hasattr(forest, "trees_") or not forest.trees_:
        raise ValueError("source forest must have fitted trees_")

    from merlin._core import _to_list

    n_trees = len(forest.trees_)
    if n_trees == 0:
        raise ValueError("forest has no fitted trees")

    X_list = _to_list(X)
    n_samples = len(X_list)

    # First pass: determine max leaves per tree across all trees
    leaves_per_tree = []
    for tree in forest.trees_:
        nl = _count_leaves(tree.root)
        leaves_per_tree.append(max(1, nl))

    max_leaves = max(leaves_per_tree) if leaves_per_tree else 1
    n_features = n_trees * max_leaves

    # Second pass: build the embedding
    from lib.array import empty

    embedding = empty((n_samples, n_features), "float64")
    for si, x in enumerate(X_list):
        for ti, tree in enumerate(forest.trees_):
            leaf_idx, _ = _sample_to_leaf_index(tree.root, x)
            col_offset = ti * max_leaves
            if 0 <= leaf_idx < max_leaves:
                embedding[si, col_offset + leaf_idx] = 1.0

    return embedding
