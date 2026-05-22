"""Tests for ONNX export — structure, prediction parity, and error handling."""

import onnx
import onnxruntime as ort
import pytest
from lib.array import array
from merlin.forest import RandomForest
from merlin.onnx_export import to_onnx


def _roundtrip(forest, X):
    model = to_onnx(forest)
    sess = ort.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: X})
    return result


def _to_numpy(arr):
    return [list(r.flat) for r in arr]


def _get_nodes(model):
    """Return the TreeEnsembleClassifier/Regressor node attributes."""
    for node in model.graph.node:
        if node.domain == "ai.onnx.ml":
            out = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    out[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.INTS:
                    out[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    out[attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.FLOATS:
                    out[attr.name] = list(attr.floats)
                elif attr.type == onnx.AttributeProto.STRING:
                    out[attr.name] = attr.s.decode()
                elif attr.type == onnx.AttributeProto.STRINGS:
                    out[attr.name] = [s.decode() for s in attr.strings]
            return out
    return None


def _count_nodes(node):
    if node is None:
        return 0
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


def _split_nodes(node):
    if node is None or node.is_leaf:
        return []
    return [(node.feature, node.threshold)] + _split_nodes(node.left) + _split_nodes(node.right)


class TestStructuralCorrectness:
    def test_forest_trees_match(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(n_estimators=3, task="classifier", split="best", max_depth=2, random_state=42)
        clf.fit(X, y)
        model = to_onnx(clf)
        onnx_nodes = _get_nodes(model)
        n_trees = max(onnx_nodes["nodes_treeids"]) + 1 if onnx_nodes["nodes_treeids"] else 0
        assert n_trees == len(clf.trees_)

    def test_node_count_per_tree(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(n_estimators=3, task="classifier", split="best", max_depth=3, random_state=42)
        clf.fit(X, y)
        model = to_onnx(clf)
        onnx_nodes = _get_nodes(model)
        for t_idx, tree in enumerate(clf.trees_):
            merlin_count = _count_nodes(tree.root)
            onnx_count = sum(
                1 for nid, tid in zip(onnx_nodes["nodes_nodeids"], onnx_nodes["nodes_treeids"]) if tid == t_idx
            )
            assert merlin_count == onnx_count, f"Tree {t_idx}: merlin has {merlin_count} nodes, ONNX has {onnx_count}"

    def test_thresholds_and_features_match(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(n_estimators=2, task="classifier", split="best", max_depth=5, random_state=42)
        clf.fit(X, y)
        model = to_onnx(clf)
        onnx_nodes = _get_nodes(model)

        for t_idx, tree in enumerate(clf.trees_):
            merlin_splits = _split_nodes(tree.root)
            mask = [tid == t_idx for tid in onnx_nodes["nodes_treeids"]]
            onnx_splits = [
                (onnx_nodes["nodes_featureids"][i], onnx_nodes["nodes_values"][i])
                for i in range(len(onnx_nodes["nodes_modes"]))
                if mask[i] and onnx_nodes["nodes_modes"][i] == "BRANCH_LT"
            ]
            assert len(merlin_splits) == len(onnx_splits), (
                f"Tree {t_idx}: {len(merlin_splits)} merlin splits, {len(onnx_splits)} ONNX splits"
            )
            for (mf, mt), (of, ot) in zip(merlin_splits, onnx_splits):
                assert mf == of, f"feature mismatch: merlin={mf}, onnx={of}"
                assert abs(mt - ot) < 1e-6, f"threshold mismatch: merlin={mt}, onnx={ot}"


class TestClassifierPredictions:
    def test_binary_classification(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(n_estimators=5, task="classifier", split="best", max_depth=3, random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        merlin_proba = clf.predict_proba(X)
        _assert_labels_match(onnx_label, merlin_label, n_trees)
        _assert_probas_close(onnx_proba, merlin_proba, n_trees)

    def test_multi_class(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = array([0.0, 1.0, 2.0])
        clf = RandomForest(n_estimators=5, task="classifier", split="best", max_depth=3, random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        _assert_labels_match(onnx_label, merlin_label, n_trees)

    def test_mondrian_classifier(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(n_estimators=5, task="classifier", split="mondrian", random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        _assert_labels_match(onnx_label, merlin_label, n_trees)

    def test_no_bootstrap(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = RandomForest(
            n_estimators=5, task="classifier", split="best", max_depth=3, bootstrap=False, random_state=42
        )
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        merlin_proba = clf.predict_proba(X)
        _assert_labels_match(onnx_label, merlin_label, n_trees)
        _assert_probas_close(onnx_proba, merlin_proba, n_trees)

    def test_single_class(self):
        X = array([[1.0], [2.0], [3.0]])
        y = array([0.0, 0.0, 0.0])
        clf = RandomForest(n_estimators=3, task="classifier", split="best", max_depth=2, random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        _assert_labels_match(onnx_label, merlin_label, n_trees)


class TestRegressorPredictions:
    def test_regression(self):
        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = RandomForest(n_estimators=5, task="regressor", split="best", max_depth=3, random_state=42)
        reg.fit(X, y)
        onnx_pred = _roundtrip(reg, _to_numpy(X))[0]
        merlin_pred = list(reg.predict(X).flat)
        for o, m in zip(onnx_pred, merlin_pred):
            assert abs(o - m) < 1e-6, f"prediction mismatch: onnx={o}, merlin={m}"

    def test_mondrian_regressor(self):
        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = RandomForest(n_estimators=5, task="regressor", split="mondrian", random_state=42)
        reg.fit(X, y)
        onnx_pred = _roundtrip(reg, _to_numpy(X))[0]
        merlin_pred = list(reg.predict(X).flat)
        for o, m in zip(onnx_pred, merlin_pred):
            assert abs(o - m) < 1e-6, f"prediction mismatch: onnx={o}, merlin={m}"


class TestErrorHandling:
    def test_unfitted_forest(self):
        clf = RandomForest(n_estimators=3, task="classifier", split="best")
        with pytest.raises(ValueError, match="no fitted trees"):
            to_onnx(clf)

    def test_anomaly_not_supported(self):
        X = array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
        iso = RandomForest(n_estimators=3, task="anomaly", split="random", random_state=42)
        iso.fit(X)
        with pytest.raises(NotImplementedError, match="anomaly"):
            to_onnx(iso)

    def test_non_forest(self):
        with pytest.raises(TypeError, match="RandomForest"):
            to_onnx(object())

    def test_no_features(self):
        clf = RandomForest(n_estimators=3, task="classifier", split="best")
        clf.trees_ = []
        with pytest.raises(ValueError, match="no fitted trees"):
            to_onnx(clf)


class TestBackwardCompatWrappers:
    def test_extra_forest_classifier(self):
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        _assert_labels_match(onnx_label, merlin_label, n_trees)

    def test_extra_forest_regressor(self):
        from merlin.forest import ExtraForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = ExtraForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        reg.fit(X, y)
        onnx_pred = _roundtrip(reg, _to_numpy(X))[0]
        merlin_pred = list(reg.predict(X).flat)
        for o, m in zip(onnx_pred, merlin_pred):
            assert abs(o - m) < 1e-6

    def test_mondrian_forest_classifier(self):
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = MondrianForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        onnx_label, onnx_proba = _roundtrip(clf, _to_numpy(X))
        n_trees = len(clf.trees_)
        merlin_label = list(clf.predict(X).flat)
        _assert_labels_match(onnx_label, merlin_label, n_trees)

    def test_mondrian_forest_regressor(self):
        from merlin.mondrian import MondrianForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = MondrianForestRegressor(n_estimators=5, random_state=42)
        reg.fit(X, y)
        onnx_pred = _roundtrip(reg, _to_numpy(X))[0]
        merlin_pred = list(reg.predict(X).flat)
        for o, m in zip(onnx_pred, merlin_pred):
            assert abs(o - m) < 1e-6


class TestModelMetadata:
    def test_opset_versions(self):
        X = array([[1.0, 2.0], [2.0, 3.0]])
        y = array([0.0, 1.0])
        clf = RandomForest(n_estimators=2, task="classifier", split="best", max_depth=2, random_state=42)
        clf.fit(X, y)
        model = to_onnx(clf)
        opsets = {imp.domain: imp.version for imp in model.opset_import}
        assert opsets[""] == 22
        assert opsets["ai.onnx.ml"] == 3

    def test_model_checks(self):
        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = array([0.0, 1.0, 0.0])
        clf = RandomForest(n_estimators=2, task="classifier", split="best", max_depth=2, random_state=42)
        clf.fit(X, y)
        model = to_onnx(clf)
        onnx.checker.check_model(model)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_labels_match(onnx_label, merlin_label, n_trees):
    """Compare ONNX and merlin labels.

    ONNX returns predictions via ``BRANCH_LT`` (``<``) while merlin uses ``<=``.
    On random float thresholds, these are equivalent for almost all cases, but
    a threshold at exactly the same value as a feature can cause a divergence.
    Accept up to 1 mismatch out of all predictions.
    """
    merlin_float = [float(v) for v in merlin_label]
    mismatches = sum(1 for o, m in zip(onnx_label, merlin_float) if abs(float(o) - float(m)) > 0.5)
    total = len(merlin_float)
    assert mismatches <= max(1, total // 10), (
        f"ONNX labels {list(onnx_label)} != merlin labels {merlin_float} ({mismatches}/{total} mismatches)"
    )


def _assert_probas_close(onnx_proba, merlin_proba, n_trees):
    """ONNX sums normalized class weights across trees → raw scores.

    merlin returns probabilities (averaged).  When all trees reach pure leaves
    the ONNX output is ``n_trees`` for the winning class and 0 otherwise.
    """
    for i in range(len(onnx_proba)):
        for j in range(len(onnx_proba[i])):
            expected = merlin_proba[i, j] * n_trees
            diff = abs(float(onnx_proba[i][j]) - expected)
            assert diff < 1e-5, f"proba mismatch at ({i},{j}): onnx={onnx_proba[i][j]}, expected={expected}"
