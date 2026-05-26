"""Tests for merlin.automl -- successive halving AutoML."""


class TestTrainTestSplit:
    def test_split_sizes(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving

        X = array([[i, i + 1] for i in range(100)])
        y = array([i % 2 for i in range(100)])

        X_train, X_val, y_train, y_val = SuccessiveHalving.train_test_split(X, y, test_size=0.25, random_state=42)
        assert len(X_train) == 75
        assert len(X_val) == 25
        assert len(y_train) == 75
        assert len(y_val) == 25

    def test_split_no_overlap(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving

        X = array([[i, i + 1] for i in range(50)])
        y = array([i % 3 for i in range(50)])

        _, X_val, _, _ = SuccessiveHalving.train_test_split(X, y, test_size=0.2, random_state=42)

    def test_split_reproducible(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving

        X = array([[i] for i in range(20)])
        y = array([i % 2 for i in range(20)])

        _, _, y1, _ = SuccessiveHalving.train_test_split(X, y, random_state=42)
        _, _, y2, _ = SuccessiveHalving.train_test_split(X, y, random_state=42)
        assert list(y1.flat) == list(y2.flat)

    def test_custom_test_size(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving

        X = array([[i] for i in range(80)])
        y = array([i % 2 for i in range(80)])

        _, X_val, _, _ = SuccessiveHalving.train_test_split(X, y, test_size=0.5, random_state=42)
        assert len(X_val) == 40


class TestSuccessiveHalvingClassifier:
    def test_fit_predict(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        X = array([[i, i + 1] for i in range(80)])
        y = array([i % 2 for i in range(80)])

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={
                "max_depth": [None, 3],
                "bootstrap": [True],
            },
            n_configs=4,
            max_resources=10,
            random_state=42,
        )
        sh.fit(X, y)

        assert sh.best_config_ is not None
        assert sh.model_ is not None
        preds = sh.predict(array([[1.0, 2.0], [3.0, 4.0]]))
        assert preds.shape == (2,)

    def test_predict_proba_shape(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        X = array([[i] for i in range(60)])
        y = array([i % 3 for i in range(60)])

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={"max_depth": [None]},
            n_configs=2,
            max_resources=8,
            random_state=42,
        )
        sh.fit(X, y)

        proba = sh.predict_proba(array([[1.0], [2.0]]))
        assert proba.shape[0] == 2
        assert proba.shape[1] == 3


class TestSuccessiveHalvingRegressor:
    def test_fit_predict(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestRegressor

        X = array([[i] for i in range(80)])
        y = array([2.0 * i + 1.0 for i in range(80)])

        sh = SuccessiveHalving(
            estimator=ExtraForestRegressor(),
            param_grid={
                "max_depth": [None, 3],
                "bootstrap": [True],
            },
            n_configs=4,
            max_resources=10,
            random_state=42,
        )
        sh.fit(X, y)

        assert sh.best_config_ is not None
        assert sh.model_ is not None
        preds = sh.predict(array([[5.0], [10.0]]))
        assert preds.shape == (2,)


class TestHistory:
    def test_history_has_rounds(self):
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        X = array([[i] for i in range(60)])
        y = array([i % 2 for i in range(60)])

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={"max_depth": [None]},
            n_configs=8,
            max_resources=10,
            random_state=42,
        )
        sh.fit(X, y)

        assert len(sh.history_) >= 2
        for round_info in sh.history_:
            assert "round" in round_info
            assert "resources" in round_info
            assert "results" in round_info
            assert len(round_info["results"]) > 0


class TestPredictBeforeFit:
    def test_predict_raises(self):
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        sh = SuccessiveHalving(estimator=ExtraForestClassifier())
        try:
            sh.predict([[1.0, 2.0]])
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_predict_proba_raises(self):
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        sh = SuccessiveHalving(estimator=ExtraForestClassifier())
        try:
            sh.predict_proba([[1.0, 2.0]])
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


class TestEstimatorRequired:
    def test_no_estimator_raises(self):
        from merlin.automl import SuccessiveHalving

        try:
            SuccessiveHalving()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "estimator must be provided" in str(e)


class TestAutoMinResources:
    def test_auto_min_resources(self):
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={"max_depth": [None]},
            n_configs=8,
            max_resources=32,
            min_resources="auto",
            random_state=42,
        )
        resources = sh._compute_rounds()
        assert resources[0] >= 1
        assert resources[-1] == 32


class TestResourceParam:
    def test_resource_override(self):
        """Verify that the resource param is overridden each round."""
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        X = array([[i] for i in range(40)])
        y = array([i % 2 for i in range(40)])

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={"max_depth": [None]},
            n_configs=4,
            max_resources=16,
            min_resources=2,
            random_state=42,
        )
        sh.fit(X, y)

        # Check that resource param appears in config keys
        assert sh.resource == "n_estimators"
        for round_info in sh.history_:
            for result in round_info["results"]:
                assert sh.resource in result["config"]


class TestCustomResource:
    def test_custom_resource_name(self):
        """Test with a different resource parameter name."""
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import RandomForest

        X = array([[i] for i in range(40)])
        y = array([i % 2 for i in range(40)])

        sh = SuccessiveHalving(
            estimator=RandomForest(task="classifier"),
            param_grid={"max_depth": [None, 3]},
            n_configs=4,
            max_resources=10,
            min_resources=2,
            random_state=42,
        )
        sh.fit(X, y)

        assert sh.best_config_ is not None
        assert sh.model_ is not None


class TestBestConfigContainsResource:
    def test_best_config_has_max_resource(self):
        """The best config should have the resource set to max_resources."""
        from lib.array import array
        from merlin.automl import SuccessiveHalving
        from merlin.forest import ExtraForestClassifier

        X = array([[i] for i in range(40)])
        y = array([i % 2 for i in range(40)])

        sh = SuccessiveHalving(
            estimator=ExtraForestClassifier(),
            param_grid={"max_depth": [None]},
            n_configs=4,
            max_resources=16,
            min_resources=2,
            random_state=42,
        )
        sh.fit(X, y)

        assert sh.best_config_[sh.resource] == 16
