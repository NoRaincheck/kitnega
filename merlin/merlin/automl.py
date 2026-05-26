"""Successive halving hyperparameter tuning.

Wraps around any estimator with a ``.fit()`` method and successively eliminates
poor configurations while increasing the resource budget for survivors. Uses a
train/validation split (no k-fold). Each configuration is evaluated once per round.

Based on: "Successive Halving: A Simple Method for Hyperparameter Optimization"
(Deforche et al., 2018; Jamieson & Talwalkar, 2016).
"""

import math
import random

from lib.array import array

from merlin._core import _to_flat_list, _to_list


def _accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true.flat, y_pred.flat) if a == b)
    return correct / len(y_true) if y_true.size > 0 else 0.0


def _mse(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true.flat, y_pred.flat)) / n


class SuccessiveHalving:
    """Successive halving hyperparameter tuner.

    Wraps around any estimator that supports ``.fit(X, y, **kwargs)`` and
    successively eliminates poor configurations while increasing the resource
    budget for survivors.

    Parameters
    ----------
    estimator : object
        Estimator instance to wrap. Must support ``.fit(X, y)``,
        ``.predict(X)``, and optionally ``.predict_proba(X)``.
    param_grid : dict of lists
        Hyperparameter search space (excluding the resource parameter).
        Keys are estimator constructor kwarg names, values are candidate lists.
    n_configs : int
        Total number of random configurations to sample.
    max_resources : int
        Max value for the *resource* parameter in the final round.
    min_resources : int or 'auto'
        Min resource value per config in the first round. If 'auto', derived
        from ``max_resources`` and ``n_configs`` so that configs fit.
    resource : str
        Name of the hyperparameter that represents the "budget" (e.g.
        ``"n_estimators"``). This param is overridden each round with the
        current resource budget. Default: ``"n_estimators"``.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    best_config_ : dict
        Best hyperparameter configuration found (resource param set to
        ``max_resources``).
    best_score_ : float
        Validation score of the best config (higher is better).
    history_ : list[dict]
        Per-round results with scores and remaining configs.
    model_ : object
        The fitted best estimator (trained on full data after tuning).
    """

    def __init__(
        self,
        estimator=None,
        param_grid=None,
        n_configs=64,
        max_resources=100,
        min_resources="auto",
        resource="n_estimators",
        random_state=None,
    ):
        if estimator is None:
            raise ValueError(
                "estimator must be provided. Pass an instance, e.g.\n"
                "  SuccessiveHalving(ExtraForestClassifier(), ...)"
            )
        self.estimator = estimator
        self.param_grid = param_grid or {}
        self.n_configs = n_configs
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.resource = resource
        self.random_state = random_state

        self.best_config_ = None
        self.best_score_ = float("-inf")
        self.history_ = []

    def _compute_rounds(self):
        """Compute number of rounds and resources per round.

        Resources double each round (geometric schedule). The first round
        uses min_resources, the last uses max_resources. Number of configs
        halved each round.
        """
        if self.min_resources == "auto":
            eta = 2
            k = int(math.log(self.max_resources) / math.log(eta)) if self.max_resources > 0 else 1
            min_res = max(1, self.max_resources // (eta**k))
        else:
            min_res = self.min_resources

        resources = []
        r = min_res
        while True:
            resources.append(r)
            if r >= self.max_resources:
                break
            next_r = r * 2
            if next_r > self.max_resources:
                resources[-1] = self.max_resources
                break
            r = next_r

        return resources

    def _sample_configs(self, rng):
        """Sample n_configs random configurations from param_grid."""
        keys = sorted(self.param_grid.keys())
        configs = []
        for _ in range(self.n_configs):
            config = {}
            for k in keys:
                candidates = self.param_grid[k]
                config[k] = rng.choice(candidates)
            configs.append(config)
        return configs

    def _score(self, y_true, y_pred):
        """Compute score: accuracy for classifiers, -MSE for regressors."""
        try:
            classes = set(y_true.flat)
            if len(classes) <= 2 and classes.issubset({0, 1}):
                return _accuracy(y_true, y_pred)
        except (AttributeError, TypeError):
            pass
        # Fall back: check if estimator has predict_proba to infer task
        if hasattr(self.estimator, "predict_proba"):
            return _accuracy(y_true, y_pred)
        return -_mse(y_true, y_pred)

    @staticmethod
    def train_test_split(X, y, test_size=0.25, random_state=None):
        """Simple train/validation split (no k-fold).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.
        test_size : float
            Fraction of data for validation split. Default 0.25.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        X_train, X_val, y_train, y_val : ndarray tuples
        """
        rng = random.Random(random_state)
        n = len(X) if hasattr(X, "__len__") else 0
        indices = list(range(n))
        rng.shuffle(indices)

        val_size = max(1, int(n * test_size))
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]

        X_list = _to_list(X)
        y_list = _to_flat_list(y)

        X_train = array([X_list[i] for i in train_idx])
        y_train = array([y_list[i] for i in train_idx])
        X_val = array([X_list[i] for i in val_idx])
        y_val = array([y_list[i] for i in val_idx])

        return X_train, X_val, y_train, y_val

    def _clone_and_update(self, config):
        """Create a fresh estimator instance with updated parameters.

        If the base estimator is an instance, copies its type and merges
        *config* on top (so resource param always comes from config).
        If it's a class, instantiates it directly with *config*.
        """
        est = self.estimator
        if isinstance(est, type):
            return est(**config)
        # Copy instance: reconstruct via constructor with merged params
        cls = type(est)
        return cls(**config)

    def fit(self, X, y):
        """Run successive halving and train best model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.

        Returns
        -------
        self
        """
        rng = random.Random(self.random_state)

        # Train/validation split
        X_train, X_val, y_train, y_val = self.train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )

        resources = self._compute_rounds()
        configs = self._sample_configs(rng)

        n_remaining = len(configs)
        round_idx = 0

        while n_remaining > 1:
            eta = 2  # halving factor
            current_resources = resources[min(round_idx, len(resources) - 1)]

            if current_resources >= self.max_resources and n_remaining <= 2:
                break

            results = []
            for i in range(n_remaining):
                config = dict(configs[i])
                # Override the resource parameter with the budget for this round
                config[self.resource] = current_resources
                if "random_state" not in config:
                    config["random_state"] = rng.randint(0, 2**31 - 1)

                model = self._clone_and_update(config)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._score(y_val, preds)

                results.append((i, config, score))

            results.sort(key=lambda r: (-r[2], rng.random()))

            self.history_.append({
                "round": round_idx,
                "resources": current_resources,
                "results": [{"config": r[1], "score": float(r[2])} for r in results],
            })

            n_keep = max(1, n_remaining // eta)
            configs = [results[i][1] for i in range(n_keep)]
            n_remaining = n_keep
            round_idx += 1

        # Final round: train all remaining at max resources
        final_resources = self.max_resources
        final_results = []
        for config in configs:
            est_config = dict(config)
            est_config[self.resource] = final_resources
            if "random_state" not in est_config:
                est_config["random_state"] = rng.randint(0, 2**31 - 1)

            model = self._clone_and_update(est_config)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = self._score(y_val, preds)

            final_results.append((config, score))

        final_results.sort(key=lambda r: (-r[1], rng.random()))

        self.history_.append({
            "round": round_idx,
            "resources": final_resources,
            "results": [{"config": r[0], "score": float(r[1])} for r in final_results],
        })

        # Train best config on full data (train + val) with max resources
        best_config = final_results[0][0]
        best_score = final_results[0][1]
        self.best_config_ = dict(best_config)
        self.best_config_[self.resource] = self.max_resources
        if "random_state" not in self.best_config_:
            self.best_config_["random_state"] = rng.randint(0, 2**31 - 1)

        # Combine train + val for final model training
        n_feat = X_train.shape[1]
        all_x_flat = list(X_train.flat) + list(X_val.flat)
        X_full = array([all_x_flat[i:i+n_feat] for i in range(0, len(all_x_flat), n_feat)])

        all_y_flat = list(y_train.flat) + list(y_val.flat)
        y_full = array(all_y_flat)

        best_model = self._clone_and_update(self.best_config_)
        best_model.fit(X_full, y_full)
        self.best_score_ = best_score
        self.model_ = best_model

        return self

    def predict(self, X):
        """Predict using the best model found."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Must call fit() before predict()")
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the best model (classifier only)."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Must call fit() before predict_proba()")
        return self.model_.predict_proba(X)
