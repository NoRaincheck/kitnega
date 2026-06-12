"""Tests for merlin — histogram-based gradient boosting."""


class TestBinning:
    def test_uniform_binning(self):
        from merlin.histogram import _bin_uniform

        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        edges, indices = _bin_uniform(x, 4)
        assert len(edges) == 5
        assert len(indices) == 5
        assert all(0 <= b < 4 for b in indices)
        assert edges[0] == 0.0
        assert edges[-1] == 4.0

    def test_quantile_binning(self):
        from merlin.histogram import _bin_quantile

        x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        edges, indices = _bin_quantile(x, 5)
        assert len(indices) == 10
        assert all(0 <= b < len(edges) - 1 for b in indices)

    def test_uniform_single_value(self):
        from merlin.histogram import _bin_uniform

        x = [3.0, 3.0, 3.0]
        edges, indices = _bin_uniform(x, 4)
        assert indices == [0, 0, 0]

    def test_quantile_single_value(self):
        from merlin.histogram import _bin_quantile

        x = [5.0, 5.0, 5.0]
        edges, indices = _bin_quantile(x, 4)
        assert indices == [0, 0, 0]


class TestHistogramClassifier:
    def test_fit_predict_binary(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]])
        y = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        clf = HistogramGradientBoosting(
            task="classifier",
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_predict_proba_shape(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])
        y = array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        clf = HistogramGradientBoosting(
            task="classifier",
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (6, 2)
        for i in range(6):
            assert abs(proba[i, 0] + proba[i, 1] - 1.0) < 1e-10

    def test_multiclass(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        clf = HistogramGradientBoosting(
            task="classifier",
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape[0] == 6
        for v in preds.flat:
            assert v in [0.0, 1.0, 2.0]

    def test_single_class(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0], [2.0], [3.0]])
        y = array([0.0, 0.0, 0.0])
        clf = HistogramGradientBoosting(
            task="classifier",
            n_estimators=5,
            max_depth=2,
            min_samples_leaf=1,
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 0.0]

    def test_quantile_binning(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]])
        y = array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        clf = HistogramGradientBoosting(
            task="classifier",
            n_estimators=10,
            max_depth=3,
            min_samples_leaf=1,
            bin_strategy="quantile",
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds.flat) == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_deterministic(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y = array([0.0, 0.0, 1.0, 1.0])
        clf1 = HistogramGradientBoosting(
            n_estimators=5,
            max_depth=2,
            min_samples_leaf=1,
            random_state=42,
        )
        clf1.fit(X, y)
        clf2 = HistogramGradientBoosting(
            n_estimators=5,
            max_depth=2,
            min_samples_leaf=1,
            random_state=42,
        )
        clf2.fit(X, y)
        assert list(clf1.predict(X).flat) == list(clf2.predict(X).flat)


class TestHistogramRegressor:
    def test_fit_predict(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        reg = HistogramGradientBoosting(
            task="regressor",
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=1,
            random_state=42,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape[0] == 6
        for i in range(6):
            assert abs(preds[i] - y[i]) < 1.0

    def test_prediction_shape(self):
        from lib.array import array
        from merlin.histogram import HistogramGradientBoosting

        X = array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = array([1.0, 2.0, 3.0])
        reg = HistogramGradientBoosting(
            task="regressor",
            n_estimators=5,
            max_depth=2,
            min_samples_leaf=1,
            random_state=42,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (3,)
