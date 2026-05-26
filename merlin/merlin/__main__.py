from merlin.forest import (
    ExtraForestClassifier,
    ExtraForestRegressor,
    IsolationForest,
    MondrianForestClassifier,
    MondrianForestRegressor,
    RandomForest,
)


def main():
    print("Merlin 0.1.0 — Random-split forests")
    print()
    print("Primary API:")
    print(f"  {RandomForest.__name__}(task=…, split=…)")
    print()
    print("  task:   'classifier' | 'regressor' | 'anomaly'")
    print("  split:  'best'      | 'random'    | 'mondrian'")
    print()
    print("Backward-compatible wrappers:")
    print(f"  {ExtraForestClassifier.__name__}")
    print(f"  {ExtraForestRegressor.__name__}")
    print(f"  {IsolationForest.__name__}")
    print(f"  {MondrianForestClassifier.__name__}")
    print(f"  {MondrianForestRegressor.__name__}")
    print()
    print("AutoML:")
    from merlin.automl import SuccessiveHalving

    print(f"  {SuccessiveHalving.__name__}(estimator=…, param_grid=…)")
    print()
    print("  Successive halving — wraps any estimator, only .fit() called.")


if __name__ == "__main__":
    main()
