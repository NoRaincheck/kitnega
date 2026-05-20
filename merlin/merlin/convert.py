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
