"""Benchmark: merlin HistogramGradientBoosting vs sklearn HistGradientBoosting vs LightGBM.

Compares classification and regression performance, training time, and prediction time
across multiple datasets and random seeds.

Usage:
    uv run python scripts/bench_histogram.py
    uv run python scripts/bench_histogram.py --seeds 3 --datasets iris diabetes

Results (2 seeds, 64 bins):

| Dataset                  | Task  | Merlin (uniform)   | Merlin (quantile)  | sklearn            | LightGBM           |
|--------------------------|-------|--------------------|--------------------|--------------------|--------------------|
| iris (150x4)             | cls   | acc=1.000          | acc=1.000          | acc=1.000          | acc=0.933          |
| breast_cancer (569x30)   | cls   | acc=0.956          | acc=0.956          | acc=0.956          | acc=0.974          |
| wine (178x13)            | cls   | acc=0.972          | acc=0.944          | acc=0.972          | acc=0.972          |
| diabetes (442x10)        | reg   | mse=3272           | mse=3159           | mse=3367           | mse=3138           |
| california_housing (20640x8) | reg | mse=0.315        | mse=0.239          | mse=0.246          | mse=0.238          |
| pumpkin_seeds (2500x12)  | cls   | acc=0.864          | acc=0.864          | acc=0.874          | acc=0.862          |

Average train time: Merlin uniform 2.35s | Merlin quantile 2.36s | sklearn 5.15s | LightGBM 1.55s

Key takeaways:
- Merlin quantile binning matches sklearn/lightgbm accuracy on regression
- Merlin trains ~2.2x faster than sklearn
- Merlin is ~3.5x slower than LightGBM (C++ vs pure Python)
- Merlin prediction is ~3x faster than sklearn
"""

import argparse
import time
from time import process_time

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import openml
except ImportError:
    openml = None

OPENML_DATASETS = {
    "46951": (46951, "classification"),
    "pumpkin_seeds": (46951, "classification"),
}

PREPROCESSORS = {
    "classification": None,
    "regression": None,
}


def _fetch_openml(data_id):
    import numpy as np

    dataset = openml.datasets.get_dataset(data_id, download_data=True)
    X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    X_np = X.to_numpy(dtype=np.float64, na_value=0.0)
    if y.dtype == "object" or y.dtype.name == "category":
        classes = sorted(y.dropna().unique())
        class_map = {c: i for i, c in enumerate(classes)}
        y_np = y.map(class_map).to_numpy(dtype=np.float64)
    else:
        y_np = y.to_numpy(dtype=np.float64, na_value=0.0)
    return X_np, y_np


def prepare_dataset(name):
    loaders = {
        "iris": (load_iris, "classification"),
        "wine": (load_wine, "classification"),
        "breast_cancer": (load_breast_cancer, "classification"),
        "diabetes": (load_diabetes, "regression"),
        "california_housing": (fetch_california_housing, "regression"),
    }

    if name in loaders:
        loader, task = loaders[name]
        data = loader()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test, task

    if name in OPENML_DATASETS:
        data_id, task = OPENML_DATASETS[name]
        if openml is None:
            raise ImportError("openml is required for OpenML datasets")
        X, y = _fetch_openml(data_id)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        return X_train, X_test, y_train, y_test, task

    raise ValueError(f"Unknown dataset: {name}")


def to_lib_array(X, y):
    from lib.array import array

    return array(X.tolist()), array(y.tolist())


def evaluate_classifier(name, y_test, y_pred, y_proba=None):
    acc = accuracy_score(y_test, y_pred)
    auc = None
    if y_proba is not None and len(set(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    elif y_proba is not None and len(set(y_test)) > 2:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    return {"accuracy": acc, "roc_auc": auc}


def evaluate_regressor(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return {"mse": mse}


def bench_merlin(X_train, X_test, y_train, y_test, task, seed, n_bins=64):
    from merlin.histogram import HistogramGradientBoosting

    task_type = "classifier" if task == "classification" else "regressor"
    X_tr, y_tr = to_lib_array(X_train, y_train)

    model = HistogramGradientBoosting(
        task=task_type,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=5,
        n_bins=n_bins,
        bin_strategy="uniform",
        random_state=seed,
    )

    t0 = process_time()
    wall0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = process_time() - t0
    wall_train = time.time() - wall0

    X_te = to_lib_array(X_test, y_test)[0]
    t0 = process_time()
    wall0 = time.time()
    if task_type == "classifier":
        y_proba = model.predict_proba(X_te)
        y_pred = model.predict(X_te)
        y_pred_flat = list(y_pred.flat)
    else:
        y_pred = model.predict(X_te)
        y_pred_flat = list(y_pred.flat)
        y_proba = None
    pred_time = process_time() - t0
    wall_pred = time.time() - wall0

    return y_pred_flat, y_proba, train_time, pred_time, wall_train, wall_pred


def bench_merlin_quantile(X_train, X_test, y_train, y_test, task, seed, n_bins=64):
    from merlin.histogram import HistogramGradientBoosting

    task_type = "classifier" if task == "classification" else "regressor"
    X_tr, y_tr = to_lib_array(X_train, y_train)

    model = HistogramGradientBoosting(
        task=task_type,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=5,
        n_bins=n_bins,
        bin_strategy="quantile",
        random_state=seed,
    )

    t0 = process_time()
    wall0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = process_time() - t0
    wall_train = time.time() - wall0

    X_te = to_lib_array(X_test, y_test)[0]
    t0 = process_time()
    wall0 = time.time()
    if task_type == "classifier":
        y_proba = model.predict_proba(X_te)
        y_pred = model.predict(X_te)
        y_pred_flat = list(y_pred.flat)
    else:
        y_pred = model.predict(X_te)
        y_pred_flat = list(y_pred.flat)
        y_proba = None
    pred_time = process_time() - t0
    wall_pred = time.time() - wall0

    return y_pred_flat, y_proba, train_time, pred_time, wall_train, wall_pred


def bench_sklearn(X_train, X_test, y_train, y_test, task, seed, n_bins=64):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=5,
            max_bins=n_bins,
            random_state=seed,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=5,
            max_bins=n_bins,
            random_state=seed,
        )

    t0 = process_time()
    wall0 = time.time()
    model.fit(X_train, y_train)
    train_time = process_time() - t0
    wall_train = time.time() - wall0

    t0 = process_time()
    wall0 = time.time()
    if task == "classification":
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_proba = None
    pred_time = process_time() - t0
    wall_pred = time.time() - wall0

    return list(y_pred), y_proba, train_time, pred_time, wall_train, wall_pred


def bench_lightgbm(X_train, X_test, y_train, y_test, task, seed, n_bins=64):
    from lightgbm import LGBMClassifier, LGBMRegressor

    if task == "classification":
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_samples=5,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
    else:
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_samples=5,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )

    t0 = process_time()
    wall0 = time.time()
    model.fit(X_train, y_train)
    train_time = process_time() - t0
    wall_train = time.time() - wall0

    t0 = process_time()
    wall0 = time.time()
    if task == "classification":
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_proba = None
    pred_time = process_time() - t0
    wall_pred = time.time() - wall0

    return list(y_pred), y_proba, train_time, pred_time, wall_train, wall_pred


BENCHMARKS = {
    "merlin_uniform": bench_merlin,
    "merlin_quantile": bench_merlin_quantile,
    "sklearn": bench_sklearn,
    "lightgbm": bench_lightgbm,
}


def run_benchmark(datasets, seeds, n_bins):
    results = []

    for ds_name in datasets:
        X_train, X_test, y_train, y_test, task = prepare_dataset(ds_name)
        n_train = len(X_train)
        n_test = len(X_test)
        n_features = X_train.shape[1]

        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} ({task})  train={n_train} test={n_test} features={n_features}")
        print(f"{'=' * 60}")

        for bench_name, bench_fn in BENCHMARKS.items():
            accs, aucs, mses = [], [], []
            train_times, pred_times = [], []
            wall_trains, wall_preds = [], []

            for seed in seeds:
                try:
                    y_pred, y_proba, tt, pt, wtr, wpd = bench_fn(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        task,
                        seed,
                        n_bins,
                    )
                    train_times.append(tt)
                    pred_times.append(pt)
                    wall_trains.append(wtr)
                    wall_preds.append(wpd)

                    if task == "classification":
                        m = evaluate_classifier(ds_name, y_test, y_pred, y_proba)
                        accs.append(m["accuracy"])
                        if m["roc_auc"] is not None:
                            aucs.append(m["roc_auc"])
                    else:
                        m = evaluate_regressor(ds_name, y_test, y_pred)
                        mses.append(m["mse"])
                except Exception as e:
                    print(f"  {bench_name} seed={seed}: ERROR {e}")

            row = {
                "dataset": ds_name,
                "task": task,
                "method": bench_name,
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "seeds": len(seeds),
            }

            if task == "classification":
                row["accuracy_mean"] = sum(accs) / len(accs) if accs else None
                row["roc_auc_mean"] = sum(aucs) / len(aucs) if aucs else None
                if row["accuracy_mean"] is not None:
                    metric_str = f"acc={row['accuracy_mean']:.4f}"
                    if row["roc_auc_mean"] is not None:
                        metric_str += f"  auc={row['roc_auc_mean']:.4f}"
                else:
                    metric_str = "FAILED"
            else:
                row["mse_mean"] = sum(mses) / len(mses) if mses else None
                if row["mse_mean"] is not None:
                    metric_str = f"mse={row['mse_mean']:.4f}"
                else:
                    metric_str = "FAILED"

            row["train_time_mean"] = sum(train_times) / len(train_times) if train_times else None
            row["pred_time_mean"] = sum(pred_times) / len(pred_times) if pred_times else None
            row["wall_train_mean"] = sum(wall_trains) / len(wall_trains) if wall_trains else None
            row["wall_pred_mean"] = sum(wall_preds) / len(wall_preds) if wall_preds else None

            train_str = f"train={row['train_time_mean']:.3f}s" if row["train_time_mean"] else "train=N/A"
            pred_str = f"pred={row['pred_time_mean']:.3f}s" if row["pred_time_mean"] else "pred=N/A"

            print(f"  {bench_name:20s}  {metric_str:30s}  {train_str}  {pred_str}")
            results.append(row)

    return results


def print_summary(results):
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Method':20s} {'Avg Accuracy/AUC':20s} {'Avg Train (s)':15s} {'Avg Pred (s)':15s}")
    print("-" * 70)

    by_method = {}
    for r in results:
        m = r["method"]
        if m not in by_method:
            by_method[m] = {"acc": [], "auc": [], "mse": [], "train": [], "pred": []}
        if r["task"] == "classification":
            if r["accuracy_mean"] is not None:
                by_method[m]["acc"].append(r["accuracy_mean"])
            if r["roc_auc_mean"] is not None:
                by_method[m]["auc"].append(r["roc_auc_mean"])
        else:
            if r["mse_mean"] is not None:
                by_method[m]["mse"].append(r["mse_mean"])
        if r["train_time_mean"] is not None:
            by_method[m]["train"].append(r["train_time_mean"])
        if r["pred_time_mean"] is not None:
            by_method[m]["pred"].append(r["pred_time_mean"])

    for method, vals in by_method.items():
        acc_str = ""
        if vals["acc"]:
            acc_str += f"acc={sum(vals['acc']) / len(vals['acc']):.4f}"
        if vals["auc"]:
            acc_str += f" auc={sum(vals['auc']) / len(vals['auc']):.4f}"
        if vals["mse"]:
            acc_str += f" mse={sum(vals['mse']) / len(vals['mse']):.1f}"
        avg_train = sum(vals["train"]) / len(vals["train"]) if vals["train"] else 0
        avg_pred = sum(vals["pred"]) / len(vals["pred"]) if vals["pred"] else 0
        print(f"{method:20s} {acc_str:30s} {avg_train:15.3f} {avg_pred:15.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark histogram gradient boosting")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--n-bins", type=int, default=64, help="Number of histogram bins")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["iris", "breast_cancer", "wine", "diabetes", "california_housing", "pumpkin_seeds"],
        help="Datasets to benchmark",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    print("Merlin Histogram Gradient Boosting Benchmark")
    print(f"Seeds: {args.seeds}, Bins: {args.n_bins}")
    print(f"Datasets: {args.datasets}")

    results = run_benchmark(args.datasets, seeds, args.n_bins)
    print_summary(results)


if __name__ == "__main__":
    main()
