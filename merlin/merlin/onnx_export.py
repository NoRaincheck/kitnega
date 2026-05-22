"""Export merlin RandomForest to ONNX format.

Requires `onnx` and optionally `onnxruntime` (for runtime verification).

Usage::

    from merlin.onnx_export import to_onnx

    model = to_onnx(forest)
    onnx.save(model, "forest.onnx")
"""

from merlin.forest import RandomForest

_ONNX_ML_DOMAIN = "ai.onnx.ml"
_ML_OPSET_VERSION = 3
_STANDARD_OPSET_VERSION = 22


def to_onnx(forest, input_name="X", label_name="label", proba_name="probabilities", output_name="variable"):
    """Convert a fitted ``RandomForest`` to an ONNX model.

    Parameters
    ----------
    forest : RandomForest
        Fitted forest (any subclass of RandomForest with ``task="classifier"``
        or ``task="regressor"``).
    input_name : str
        Name for the input tensor (float, shape ``(N, n_features)``).
    label_name : str
        Name for the predicted-label output (classifier only).
    proba_name : str
        Name for the probability output (classifier only).
    output_name : str
        Name for the regression output (regressor only).

    Returns
    -------
    onnx.ModelProto
    """
    from onnx import TensorProto, helper

    if not isinstance(forest, RandomForest):
        raise TypeError(f"expected RandomForest, got {type(forest).__name__}")
    if forest.task not in ("classifier", "regressor"):
        raise NotImplementedError(f"ONNX export not implemented for task={forest.task!r}")
    if not forest.trees_:
        raise ValueError("forest has no fitted trees; call fit() first")

    n_features = forest.n_features_in_
    if n_features == 0:
        raise ValueError("forest has n_features_in_=0; was fit() called?")

    cls_nodes = []
    cls_leaf_attrs = []
    reg_nodes = []
    reg_leaf_attrs = []

    for t_idx in range(len(forest.trees_)):
        nodes, leaf_attrs = _collect_tree(forest, t_idx)
        if forest.task == "classifier":
            cls_nodes.extend(nodes)
            cls_leaf_attrs.extend(leaf_attrs)
        else:
            reg_nodes.extend(nodes)
            reg_leaf_attrs.extend(leaf_attrs)

    input_type = TensorProto.FLOAT
    input_shape = [None, n_features]
    inputs = [helper.make_tensor_value_info(input_name, input_type, input_shape)]

    if forest.task == "classifier":
        graph_nodes, graph_outputs, initializers = _build_classifier_graph(
            cls_nodes,
            cls_leaf_attrs,
            forest,
            input_name,
            label_name,
            proba_name,
        )
    else:
        graph_nodes, graph_outputs, initializers = _build_regressor_graph(
            reg_nodes,
            reg_leaf_attrs,
            input_name,
            output_name,
        )

    graph = helper.make_graph(graph_nodes, "merlin_forest", inputs, graph_outputs, initializers)

    opset_imports = [
        helper.make_opsetid("", _STANDARD_OPSET_VERSION),
        helper.make_opsetid(_ONNX_ML_DOMAIN, _ML_OPSET_VERSION),
    ]

    model = helper.make_model(graph, opset_imports=opset_imports, producer_name="merlin", producer_version="0.1.0")
    return model


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _build_classifier_graph(nodes, leaf_attrs, forest, input_name, label_name, proba_name):
    """Build the classifier ONNX graph with label-correction subgraph.

    onnxruntime 1.26.0 has a bug where ``TreeEnsembleClassifier``'s label
    output is always 1 (or the last class index).  The proba/score output is
    correct.  We work around this by post-processing the proba output with
    ``TopK → Reshape → Gather`` to produce the correct label.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    n_classes = forest.n_classes_
    classes = list(forest.classes_.flat)
    classlabels = [int(c) for c in classes]

    raw_label = f"_{label_name}_raw"

    attrs = _build_classifier_attrs(nodes, leaf_attrs, classlabels)
    # Proba output name matches directly (no correction needed for proba)
    classifier_node = helper.make_node(
        "TreeEnsembleClassifier",
        inputs=[input_name],
        outputs=[raw_label, proba_name],
        domain=_ONNX_ML_DOMAIN,
        **attrs,
    )

    graph_nodes = [classifier_node]

    # Correct label from proba: TopK + Reshape + Gather
    k = np.array([1], dtype=np.int64)
    k_node = helper.make_node(
        "Constant",
        [],
        ["_k_val"],
        value=numpy_helper.from_array(k, "_k_tensor"),
    )
    topk_node = helper.make_node(
        "TopK",
        [proba_name, "_k_val"],
        ["_top_values", "_top_indices"],
        axis=1,
        largest=1,
    )
    shape = np.array([-1], dtype=np.int64)
    shape_node = helper.make_node(
        "Constant",
        [],
        ["_shape_val"],
        value=numpy_helper.from_array(shape, "_shape_tensor"),
    )
    reshape_node = helper.make_node(
        "Reshape",
        ["_top_indices", "_shape_val"],
        ["_flat_indices"],
    )
    cls_tensor = numpy_helper.from_array(
        np.array(classlabels, dtype=np.int64),
        "classlabels",
    )
    gather_node = helper.make_node(
        "Gather",
        ["classlabels", "_flat_indices"],
        [label_name],
        axis=0,
    )

    graph_nodes.extend([k_node, topk_node, shape_node, reshape_node, gather_node])

    graph_outputs = [
        helper.make_tensor_value_info(label_name, TensorProto.INT64, [None]),
        helper.make_tensor_value_info(proba_name, TensorProto.FLOAT, [None, n_classes]),
    ]
    return graph_nodes, graph_outputs, [cls_tensor]


def _build_regressor_graph(nodes, leaf_attrs, input_name, output_name):
    """Build the regressor ONNX graph (no label-correction needed)."""
    from onnx import TensorProto, helper

    attrs = _build_regressor_attrs(nodes, leaf_attrs)
    node = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=[input_name],
        outputs=[output_name],
        domain=_ONNX_ML_DOMAIN,
        **attrs,
    )
    graph_outputs = [
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, 1]),
    ]
    return [node], graph_outputs, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_tree(forest, tree_idx):
    """Collect ONNX node/leaf attributes from a single merlin tree.

    Returns
    -------
    nodes : list of dict
        Split/leaf node descriptors with keys:
        tree_id, node_id, mode, feature, threshold, true_id, false_id
    leaf_attrs : list of tuple
        Classifier: (tree_id, node_id, class_id, weight)
        Regressor:  (tree_id, node_id, target_id, weight)
    """
    tree = forest.trees_[tree_idx]
    id_map = {}
    nodes = []
    leaf_attrs = []

    def _assign(node, counter):
        if node is None:
            return
        nid = counter[0]
        counter[0] += 1
        id_map[node] = nid
        _assign(node.left, counter)
        _assign(node.right, counter)

    _assign(tree.root, [0])

    def _collect(node):
        if node is None:
            return
        nid = id_map[node]
        if node.is_leaf:
            # Leaf children must point to a valid node id (ORT 1.26
            # validates child ids are within [0, n_nodes)).  Point to
            # node 0 (the root of this tree) since leaf children are never
            # traversed.
            nodes.append(
                {
                    "tree_id": tree_idx,
                    "node_id": nid,
                    "mode": "LEAF",
                    "feature": 0,
                    "threshold": 0.0,
                    "true_id": 0,
                    "false_id": 0,
                }
            )
            if forest.task == "classifier":
                pred = node.prediction
                if pred is None:
                    n_cls = forest.n_classes_
                    pred = [0] * n_cls
                total = sum(pred)
                if total > 0:
                    for c, count in enumerate(pred):
                        leaf_attrs.append((tree_idx, nid, c, float(count) / total))
                else:
                    n_cls = len(pred)
                    for c in range(n_cls):
                        leaf_attrs.append((tree_idx, nid, c, 1.0 / n_cls))
            else:
                leaf_attrs.append((tree_idx, nid, 0, float(node.prediction if node.prediction is not None else 0.0)))
        else:
            nodes.append(
                {
                    "tree_id": tree_idx,
                    "node_id": nid,
                    "mode": "BRANCH_LT",
                    "feature": node.feature,
                    "threshold": node.threshold,
                    "true_id": id_map[node.left],
                    "false_id": id_map[node.right],
                }
            )
            _collect(node.left)
            _collect(node.right)

    _collect(tree.root)
    return nodes, leaf_attrs


def _build_classifier_attrs(nodes, leaf_attrs, classlabels):
    """Build attribute dict for TreeEnsembleClassifier."""
    return {
        "nodes_treeids": [n["tree_id"] for n in nodes],
        "nodes_nodeids": [n["node_id"] for n in nodes],
        "nodes_featureids": [n["feature"] for n in nodes],
        "nodes_values": [n["threshold"] for n in nodes],
        "nodes_modes": [n["mode"] for n in nodes],
        "nodes_truenodeids": [n["true_id"] for n in nodes],
        "nodes_falsenodeids": [n["false_id"] for n in nodes],
        "class_treeids": [a[0] for a in leaf_attrs],
        "class_nodeids": [a[1] for a in leaf_attrs],
        "class_ids": [a[2] for a in leaf_attrs],
        "class_weights": [a[3] for a in leaf_attrs],
        "classlabels_int64s": classlabels,
        "post_transform": "NONE",
    }


def _build_regressor_attrs(nodes, leaf_attrs):
    """Build attribute dict for TreeEnsembleRegressor."""
    return {
        "nodes_treeids": [n["tree_id"] for n in nodes],
        "nodes_nodeids": [n["node_id"] for n in nodes],
        "nodes_featureids": [n["feature"] for n in nodes],
        "nodes_values": [n["threshold"] for n in nodes],
        "nodes_modes": [n["mode"] for n in nodes],
        "nodes_truenodeids": [n["true_id"] for n in nodes],
        "nodes_falsenodeids": [n["false_id"] for n in nodes],
        "target_treeids": [a[0] for a in leaf_attrs],
        "target_nodeids": [a[1] for a in leaf_attrs],
        "target_ids": [a[2] for a in leaf_attrs],
        "target_weights": [a[3] for a in leaf_attrs],
        "n_targets": 1,
        "aggregate_function": "AVERAGE",
        "post_transform": "NONE",
    }
