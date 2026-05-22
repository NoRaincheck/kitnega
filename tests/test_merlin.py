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
