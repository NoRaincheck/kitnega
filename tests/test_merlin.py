"""Tests for merlin — random-split forests."""


class TestForestClassifier:
    def test_fit_predict_binary(self):
        from lib.array import array
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 1.0, 1.0]

    def test_predict_proba_shape(self):
        from lib.array import array
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = array([0.0, 1.0, 2.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (3, 3)
        for i in range(3):
            assert abs(sum(proba[i, j] for j in range(3)) - 1.0) < 1e-10

    def test_no_bootstrap(self):
        from lib.array import array
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = ExtraForestClassifier(n_estimators=5, bootstrap=False, max_depth=3, random_state=42)
        clf.fit(X, y)
        assert len(clf.trees_) == 5
        assert list(clf.predict(X).flat) == [0.0, 0.0, 1.0, 1.0]

    def test_single_class(self):
        from lib.array import array
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0], [2.0], [3.0]])
        y = array([0.0, 0.0, 0.0])
        clf = ExtraForestClassifier(n_estimators=3, max_depth=2, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 0.0]


class TestForestRegressor:
    def test_fit_predict(self):
        from lib.array import array
        from merlin.forest import ExtraForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = ExtraForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape[0] == 4

    def test_no_bootstrap(self):
        from lib.array import array
        from merlin.forest import ExtraForestRegressor

        X = array([[1.0], [2.0]])
        y = array([1.0, 2.0])
        reg = ExtraForestRegressor(n_estimators=3, bootstrap=False, max_depth=2, random_state=42)
        reg.fit(X, y)
        assert len(reg.trees_) == 3


class TestIsolationForest:
    def test_fit_score(self):
        from lib.array import array
        from merlin.isoforest import IsolationForest

        X = array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [10.0, 10.0]])
        iso = IsolationForest(n_estimators=10, random_state=42)
        iso.fit(X)
        scores = iso.score_samples(X)
        assert scores.shape == (4,)

    def test_predict_shape(self):
        from lib.array import array
        from merlin.isoforest import IsolationForest

        X = array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
        iso = IsolationForest(n_estimators=5, random_state=42)
        iso.fit(X)
        preds = iso.predict(X)
        assert preds.shape == (3,)

    def test_decision_function(self):
        from lib.array import array
        from merlin.isoforest import IsolationForest

        X = array([[1.0, 1.0], [2.0, 2.0]])
        iso = IsolationForest(n_estimators=5, random_state=42)
        iso.fit(X)
        df = iso.decision_function(X)
        assert df.shape == (2,)


class TestMondrianClassifier:
    def test_fit_predict_binary(self):
        from lib.array import array
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = MondrianForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 1.0, 1.0]

    def test_predict_proba_shape(self):
        from lib.array import array
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0], [2.0], [3.0]])
        y = array([0.0, 1.0, 0.0])
        clf = MondrianForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (3, 2)

    def test_partial_fit(self):
        from lib.array import array
        from merlin.mondrian import MondrianForestClassifier

        X1 = array([[1.0, 2.0], [2.0, 3.0]])
        y1 = array([0.0, 0.0])
        X2 = array([[3.0, 4.0], [4.0, 5.0]])
        y2 = array([1.0, 1.0])

        clf = MondrianForestClassifier(n_estimators=5, random_state=42)
        clf.partial_fit(X1, y1, classes=[0.0, 1.0])
        clf.partial_fit(X2, y2)
        preds = clf.predict(array([[1.0, 2.0], [4.0, 5.0]]))
        assert preds.shape[0] == 2


class TestMondrianRegressor:
    def test_fit_predict(self):
        from lib.array import array
        from merlin.mondrian import MondrianForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = MondrianForestRegressor(n_estimators=5, random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape[0] == 4

    def test_partial_fit(self):
        from lib.array import array
        from merlin.mondrian import MondrianForestRegressor

        reg = MondrianForestRegressor(n_estimators=3, random_state=42)
        reg.partial_fit(array([[1.0], [2.0]]), array([1.0, 2.0]))
        reg.partial_fit(array([[3.0]]), array([3.0]))
        preds = reg.predict(array([[1.0], [3.0]]))
        assert preds.shape[0] == 2


class TestConvert:
    def test_extra_classifier_to_mondrian(self):
        from lib.array import array
        from merlin.convert import extra_to_mondrian
        from merlin.forest import ExtraForestClassifier
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y = array([0.0, 0.0, 0.0, 1.0, 1.0])
        ef = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        ef.fit(X, y)
        mf = extra_to_mondrian(ef)
        assert isinstance(mf, MondrianForestClassifier)
        assert len(mf.trees_) == len(ef.trees_)
        preds = mf.predict(X)
        assert preds.shape == (5,)

    def test_extra_regressor_to_mondrian(self):
        from lib.array import array
        from merlin.convert import extra_to_mondrian
        from merlin.forest import ExtraForestRegressor
        from merlin.mondrian import MondrianForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = array([1.0, 2.0, 3.0, 4.0, 5.0])
        ef = ExtraForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        ef.fit(X, y)
        mf = extra_to_mondrian(ef)
        assert isinstance(mf, MondrianForestRegressor)
        assert len(mf.trees_) == len(ef.trees_)
        preds = mf.predict(X)
        assert preds.shape == (5,)

    def test_mondrian_classifier_to_extra(self):
        from lib.array import array
        from merlin.convert import mondrian_to_extra
        from merlin.forest import ExtraForestClassifier
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        mf = MondrianForestClassifier(n_estimators=5, random_state=42)
        mf.fit(X, y)
        ef = mondrian_to_extra(mf)
        assert isinstance(ef, ExtraForestClassifier)
        assert len(ef.trees_) == len(mf.trees_)
        preds = ef.predict(X)
        assert preds.shape == (4,)

    def test_mondrian_regressor_to_extra(self):
        from lib.array import array
        from merlin.convert import mondrian_to_extra
        from merlin.forest import ExtraForestRegressor
        from merlin.mondrian import MondrianForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        mf = MondrianForestRegressor(n_estimators=5, random_state=42)
        mf.fit(X, y)
        ef = mondrian_to_extra(mf)
        assert isinstance(ef, ExtraForestRegressor)
        assert len(ef.trees_) == len(mf.trees_)
        preds = ef.predict(X)
        assert preds.shape == (4,)

    def test_extra_classifier_to_iso(self):
        from lib.array import array
        from merlin.convert import extra_to_iso
        from merlin.forest import ExtraForestClassifier
        from merlin.isoforest import IsolationForest

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [10.0, 10.0]])
        y = array([0.0, 0.0, 0.0, 1.0, 1.0])
        ef = ExtraForestClassifier(n_estimators=5, max_depth=4, random_state=42)
        ef.fit(X, y)
        iso = extra_to_iso(ef)
        assert isinstance(iso, IsolationForest)
        assert len(iso.trees_) == len(ef.trees_)
        preds = iso.predict(X)
        assert preds.shape == (5,)
        scores = iso.score_samples(X)
        assert scores.shape == (5,)

    def test_convert_roundtrip_classifier(self):
        from lib.array import array
        from merlin.convert import extra_to_mondrian, mondrian_to_extra
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        ef = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        ef.fit(X, y)
        mf = extra_to_mondrian(ef)
        ef2 = mondrian_to_extra(mf)
        assert len(ef2.trees_) == len(ef.trees_)
        assert ef2.n_classes_ == ef.n_classes_
        preds = ef2.predict(X)
        assert preds.shape == (4,)

    def test_convert_errors(self):
        from merlin.convert import extra_to_iso, extra_to_mondrian, mondrian_to_extra

        try:
            extra_to_mondrian(object())
            assert False, "expected TypeError"
        except TypeError:
            pass

        try:
            mondrian_to_extra(object())
            assert False, "expected TypeError"
        except TypeError:
            pass

        try:
            extra_to_iso(object())
            assert False, "expected TypeError"
        except TypeError:
            pass

    def test_forest_to_embedding_classifier(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)

        emb = forest_to_embedding(clf, X)
        assert emb.shape[0] == 4  # n_samples
        assert emb.shape[1] > 0  # some features from trees
        # Each row should have exactly one 1.0 per tree
        for i in range(emb.shape[0]):
            for ti in range(5):
                col_start = ti * emb.shape[1] // 5
                col_end = col_start + (emb.shape[1] // 5)
                row_slice = sum(emb[i, j] for j in range(col_start, min(col_end, emb.shape[1])))
                assert abs(row_slice - 1.0) < 1e-10

    def test_forest_to_embedding_regressor(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding
        from merlin.forest import ExtraForestRegressor

        X = array([[1.0], [2.0], [3.0], [4.0]])
        y = array([1.0, 2.0, 3.0, 4.0])
        reg = ExtraForestRegressor(n_estimators=5, max_depth=2, random_state=42)
        reg.fit(X, y)

        emb = forest_to_embedding(reg, X)
        assert emb.shape[0] == 4
        # Same-row-sum check per tree
        n_trees = 5
        col_per_tree = emb.shape[1] // n_trees + (1 if emb.shape[1] % n_trees else 0)
        for i in range(emb.shape[0]):
            for ti in range(n_trees):
                start = ti * col_per_tree
                end = min(start + col_per_tree, emb.shape[1])
                row_sum = sum(emb[i, j] for j in range(start, end))
                assert abs(row_sum - 1.0) < 1e-10

    def test_forest_to_embedding_mondrian(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding
        from merlin.mondrian import MondrianForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = array([0.0, 1.0, 0.0])
        clf = MondrianForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)

        emb = forest_to_embedding(clf, X)
        assert emb.shape[0] == 3
        # Verify exactly one 1.0 per tree in each row
        for i in range(emb.shape[0]):
            for ti in range(5):
                col_start = ti * (emb.shape[1] // 5)
                col_end = min(col_start + (emb.shape[1] // 5), emb.shape[1])
                row_sum = sum(emb[i, j] for j in range(col_start, col_end))
                assert abs(row_sum - 1.0) < 1e-10

    def test_forest_to_embedding_empty_forest(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding

        class EmptyForest:
            trees_ = []

        try:
            forest_to_embedding(EmptyForest(), array([[1.0]]))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_forest_to_embedding_consistency(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = array([0.0, 1.0, 0.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)

        emb1 = forest_to_embedding(clf, X)
        emb2 = forest_to_embedding(clf, X)
        # Same input should produce same embedding (deterministic for trained model)
        assert list(emb1.flat) == list(emb2.flat)

    def test_forest_to_embedding_different_samples(self):
        from lib.array import array
        from merlin.convert import forest_to_embedding
        from merlin.forest import ExtraForestClassifier

        X = array([[1.0, 2.0], [4.0, 5.0]])
        y = array([0.0, 1.0])
        clf = ExtraForestClassifier(n_estimators=5, max_depth=3, random_state=42)
        clf.fit(X, y)

        emb = forest_to_embedding(clf, X)
        # Different samples should have different embeddings (at least in some trees)
        assert list(emb[0]) != list(emb[1])


# ---------------------------------------------------------------------------
# Numerical embedding tests
# ---------------------------------------------------------------------------


class TestPiecewiseLinearEmbedding:
    def test_basic_shape(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        X = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        enc = PiecewiseLinearEncoder(n_bins=3)
        emb = enc.fit_transform(X)
        # Each feature gets len(boundaries)-1+1 output columns per feature
        assert emb.shape[0] == 3
        assert emb.shape[1] > 0

    def test_row_sums_per_feature(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        X = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        enc = PiecewiseLinearEncoder(n_bins=3)
        emb = enc.fit_transform(X)
        # Each feature's output columns sum to 1.0
        boundaries = enc.boundaries_
        for i in range(emb.shape[0]):
            col_offset = 0
            for j in range(len(boundaries)):
                stride = len(boundaries[j]) - 1 + 1
                s = sum(emb[i, k] for k in range(col_offset, col_offset + stride))
                assert abs(s - 1.0) < 1e-10
                col_offset += stride

    def test_values_at_boundaries(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        X = array([[0.0], [5.0], [10.0]])
        enc = PiecewiseLinearEncoder(n_bins=2)
        emb = enc.fit_transform(X)
        # First sample at min boundary → all weight on first bin
        assert abs(emb[0, 0] - 1.0) < 1e-10
        # Last sample at max boundary → all weight on last bin
        assert abs(emb[2, 2] - 1.0) < 1e-10

    def test_fit_then_transform(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        X_train = array([[1.0], [3.0], [5.0]])
        X_test = array([[2.0], [4.0]])
        enc = PiecewiseLinearEncoder(n_bins=3)
        emb_train = enc.fit_transform(X_train)
        emb_test = enc.transform(X_test)
        assert emb_train.shape[0] == 3
        assert emb_test.shape[0] == 2

    def test_univariate(self):
        from lib.array import array
        from merlin.embeddings import piecewise_linear_embedding

        X = array([[1.0], [2.0], [3.0]])
        emb = piecewise_linear_embedding(X, n_bins=5)
        assert emb.shape[0] == 3
        # Each row sum should be 1.0 (single feature)
        for i in range(3):
            s = sum(emb[i])
            assert abs(s - 1.0) < 1e-10

    def test_deterministic(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        X = array([[1.0, 2.0], [3.0, 4.0]])
        enc = PiecewiseLinearEncoder(n_bins=3, random_state=42)
        emb1 = enc.fit_transform(X.copy())
        enc2 = PiecewiseLinearEncoder(n_bins=3, random_state=42)
        emb2 = enc2.fit_transform(X.copy())
        assert list(emb1.flat) == list(emb2.flat)

    def test_requires_fit(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearEncoder

        enc = PiecewiseLinearEncoder()
        try:
            enc.transform(array([[1.0]]))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


class TestPiecewiseLinearForestEncoding:
    def test_basic_shape(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        X = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        enc = PiecewiseLinearForestEncoder(n_estimators=5, n_bins=3, random_state=42)
        emb = enc.fit_transform(X)
        assert emb.shape[0] == 3
        assert emb.shape[1] > 0  # some output features from tree splits

    def test_row_sums(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        X = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        enc = PiecewiseLinearForestEncoder(n_estimators=5, n_bins=3, random_state=42)
        emb = enc.fit_transform(X)
        # Each feature sums to ~1.0 per row
        boundaries = enc.boundaries_
        for i in range(emb.shape[0]):
            col_offset = 0
            for j in range(len(boundaries)):
                stride = len(boundaries[j]) - 1 + 1
                s = sum(emb[i, k] for k in range(col_offset, col_offset + stride))
                assert abs(s - 1.0) < 1e-9
                col_offset += stride

    def test_fit_then_transform(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        X_train = array([[1.0], [3.0], [5.0]])
        X_test = array([[2.0], [4.0]])
        enc = PiecewiseLinearForestEncoder(n_estimators=5, n_bins=3, random_state=42)
        emb_train = enc.fit_transform(X_train)
        emb_test = enc.transform(X_test)
        assert emb_train.shape[0] == 3
        assert emb_test.shape[0] == 2

    def test_univariate(self):
        from lib.array import array
        from merlin.embeddings import piecewise_linear_forest_embedding

        X = array([[1.0], [2.0], [3.0]])
        emb = piecewise_linear_forest_embedding(X, n_estimators=5, n_bins=5, random_state=42)
        assert emb.shape[0] == 3
        for i in range(3):
            s = sum(emb[i])
            assert abs(s - 1.0) < 1e-9

    def test_requires_fit(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        enc = PiecewiseLinearForestEncoder()
        try:
            enc.transform(array([[1.0]]))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_different_samples_produce_different_embeddings(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        X = array([[1.0], [10.0], [50.0], [100.0]])
        enc = PiecewiseLinearForestEncoder(n_estimators=10, n_bins=3, random_state=42)
        emb = enc.fit_transform(X)
        # Different samples should produce different embeddings
        assert list(emb[0]) != list(emb[1])

    def test_deterministic(self):
        from lib.array import array
        from merlin.embeddings import PiecewiseLinearForestEncoder

        X = array([[1.0, 2.0], [3.0, 4.0]])
        enc1 = PiecewiseLinearForestEncoder(n_estimators=5, n_bins=3, random_state=42)
        emb1 = enc1.fit_transform(X.copy())
        enc2 = PiecewiseLinearForestEncoder(n_estimators=5, n_bins=3, random_state=42)
        emb2 = enc2.fit_transform(X.copy())
        assert list(emb1.flat) == list(emb2.flat)


class TestNumericalEmbeddingFactory:
    def test_piecewise_linear_strategy(self):
        from lib.array import array
        from merlin.embeddings import numerical_embedding

        X = array([[1.0, 2.0], [3.0, 4.0]])
        emb = numerical_embedding(X, strategy="piecewise-linear", n_bins=3)
        assert emb.shape[0] == 2

    def test_tree_split_strategy(self):
        from lib.array import array
        from merlin.embeddings import numerical_embedding

        X = array([[1.0], [2.0], [3.0]])
        emb = numerical_embedding(
            X, strategy="tree-split", n_estimators=5, n_bins=3, random_state=42
        )
        assert emb.shape[0] == 3

    def test_invalid_strategy(self):
        from lib.array import array
        from merlin.embeddings import numerical_embedding

        try:
            numerical_embedding(array([[1.0]]), strategy="invalid")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_kwargs_passed_to_forest(self):
        from lib.array import array
        from merlin.embeddings import numerical_embedding

        X = array([[1.0], [2.0], [3.0]])
        # max_depth limits tree depth, should still work
        emb = numerical_embedding(
            X,
            strategy="tree-split",
            n_estimators=5,
            max_depth=2,
            n_bins=3,
            random_state=42,
        )
        assert emb.shape[0] == 3
