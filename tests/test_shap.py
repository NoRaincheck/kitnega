"""Tests for TreeSHAP — SHAP value computation for tree ensembles."""


class TestTreeShap:
    def test_regressor_sum_property(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        y = [1.0, 2.0, 3.0, 4.0]
        rf = RandomForest(
            task="regressor", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        preds = rf.predict(X)
        for i in range(len(X)):
            s = sum(shap[i]) + bias
            assert abs(s - preds[i]) < 1e-10

    def test_regressor_output_shap_flag(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        y = [1.0, 2.0, 3.0, 4.0]
        rf = RandomForest(
            task="regressor", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        preds1 = rf.predict(X)
        preds2, shap, bias = rf.predict(X, output_shap=True)
        for i in range(len(X)):
            assert preds1[i] == preds2[i]

    def test_classifier_sum_property(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        y = [0, 0, 1, 1, 1]
        rf = RandomForest(
            task="classifier", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        probs = rf.predict_proba(X)
        for i in range(len(X)):
            for c in range(rf.n_classes_):
                s = sum(shap[i][j][c] for j in range(rf.n_features_in_)) + bias[c]
                assert abs(s - probs[i][c]) < 1e-10

    def test_classifier_output_shap_flag(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        y = [0, 0, 1, 1, 1]
        rf = RandomForest(
            task="classifier", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        probs1 = rf.predict_proba(X)
        probs2, shap, bias = rf.predict_proba(X, output_shap=True)
        for i in range(len(X)):
            for c in range(rf.n_classes_):
                assert abs(probs1[i][c] - probs2[i][c]) < 1e-10

    def test_classifier_predict_output_shap(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        y = [0, 0, 1, 1, 1]
        rf = RandomForest(
            task="classifier", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        preds1 = rf.predict(X)
        preds2, shap, bias = rf.predict(X, output_shap=True)
        for i in range(len(X)):
            assert preds1[i] == preds2[i]

    def test_multi_class_sum_property(self):
        from merlin.forest import RandomForest

        X = [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
            [0.9, 1.0, 0.1],
        ]
        y = [0, 1, 2, 0, 1]
        rf = RandomForest(
            task="classifier", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        probs = rf.predict_proba(X)
        M = rf.n_features_in_
        for i in range(len(X)):
            for c in range(rf.n_classes_):
                s = sum(shap[i][j][c] for j in range(M)) + bias[c]
                assert abs(s - probs[i][c]) < 1e-10

    def test_single_tree(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [1.0, 2.0]
        rf = RandomForest(
            task="regressor", n_estimators=1, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        preds = rf.predict(X)
        for i in range(len(X)):
            s = sum(shap[i]) + bias
            assert abs(s - preds[i]) < 1e-10

    def test_single_sample(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        y = [1.0, 2.0, 3.0]
        rf = RandomForest(
            task="regressor", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values([X[0]])
        assert len(shap) == 1
        assert len(shap[0]) == 2
        s = sum(shap[0]) + bias
        assert abs(s - rf.predict([X[0]])[0]) < 1e-10

    def test_one_feature(self):
        from merlin.forest import RandomForest

        X = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        rf = RandomForest(
            task="regressor", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        preds = rf.predict(X)
        for i in range(len(X)):
            assert abs(sum(shap[i]) + bias - preds[i]) < 1e-10

    def test_shap_bruteforce_agreement(self):
        from merlin._shap import _shap_bruteforce, _tree_shap
        import random

        rng = random.Random(42)
        for M in range(1, 5):
            for seed in range(10):
                rng = random.Random(seed * 1000 + M)

                def _node(M, depth):
                    from merlin._core import _Node

                    if depth == 0 or rng.random() < 0.3:
                        n = _Node()
                        n.is_leaf = True
                        n.prediction = rng.uniform(-10, 10)
                        n.size = rng.randint(10, 100)
                        return n
                    n = _Node()
                    n.feature = rng.randint(0, M - 1)
                    n.threshold = rng.uniform(-1, 1)
                    n.left = _node(M, depth - 1)
                    n.right = _node(M, depth - 1)
                    n.is_leaf = False
                    n.size = n.left.size + n.right.size
                    return n

                from merlin.forest import _Tree

                tree = _Tree()
                tree.root = _node(M, 4)
                x = [rng.uniform(-1, 1) for _ in range(M)]
                phi_bf = _shap_bruteforce(tree, x, M)
                phi_ts = _tree_shap(tree, x, M)
                for i in range(M + 1):
                    assert abs(phi_bf[i] - phi_ts[i]) < 1e-8

    def test_no_trees_raises(self):
        from merlin.forest import RandomForest

        rf = RandomForest(task="regressor", n_estimators=0)
        try:
            rf.shap_values([[0.1, 0.2]])
            assert False, "should raise"
        except ValueError:
            pass

    def test_regressor_shape(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]]
        y = [1.0, 2.0, 3.0]
        rf = RandomForest(
            task="regressor", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        assert len(shap) == 3
        assert len(shap[0]) == 3
        assert isinstance(bias, float)

    def test_classifier_shape(self):
        from merlin.forest import RandomForest

        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        y = [0, 0, 1, 1, 1]
        rf = RandomForest(
            task="classifier", n_estimators=5, max_depth=3, random_state=42
        )
        rf.fit(X, y)
        shap, bias = rf.shap_values(X)
        assert len(shap) == 5
        assert len(shap[0]) == 2
        assert len(shap[0][0]) == 2
        assert isinstance(bias, list)
        assert len(bias) == 2
