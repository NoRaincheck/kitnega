"""TreeSHAP — SHAP value computation for tree ensembles.

Implements:
  Brute-force (exponential in M) for verification.
  Polynomial-time TreeSHAP (per-leaf, accounts for feature direction).
"""

import math

from merlin._core import _to_list

# ---------------------------------------------------------------------------
# Scalar expected value (regressor)
# ---------------------------------------------------------------------------


def _expected_value(node, x, known):
    if node.is_leaf:
        return node.prediction
    f = node.feature
    if f in known:
        if x[f] <= node.threshold:
            return _expected_value(node.left, x, known)
        return _expected_value(node.right, x, known)
    left_w = node.left.size / node.size
    return left_w * _expected_value(node.left, x, known) + (1 - left_w) * _expected_value(node.right, x, known)


# ---------------------------------------------------------------------------
# Brute-force SHAP (for verification only, M ≤ 12)
# ---------------------------------------------------------------------------


def _shap_bruteforce(tree, x, M):
    phi = [0.0] * (M + 1)
    for j in range(M):
        total = 0.0
        others = [i for i in range(M) if i != j]
        for mask in range(1 << (M - 1)):
            k = 0
            s_set = set()
            for bit in range(M - 1):
                if mask & (1 << bit):
                    s_set.add(others[bit])
                    k += 1
            w = math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M)
            with_j = _expected_value(tree.root, x, s_set | {j})
            without = _expected_value(tree.root, x, s_set)
            total += w * (with_j - without)
        phi[j] = total
    phi[M] = _expected_value(tree.root, x, set())
    return phi


# ---------------------------------------------------------------------------
# Polynomial-time TreeSHAP
# ---------------------------------------------------------------------------


def _weight(k, M):
    if k < 0 or k >= M:
        return 0.0
    return math.factorial(k) * math.factorial(M - k - 1) / math.factorial(M)


def _binomial(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _leaf_contributions(leaf_val, toward, away, M):
    phi = [0.0] * (M + 1)

    toward_prod = {}
    for fid, o in toward:
        toward_prod[fid] = toward_prod.get(fid, 1.0) * o

    away_prod = {}
    for fid, z in away:
        away_prod[fid] = away_prod.get(fid, 1.0) * z

    mixed = set(toward_prod) & set(away_prod)
    for fid in mixed:
        del toward_prod[fid]

    A = [(fid, toward_prod[fid]) for fid in sorted(toward_prod)]
    B = sorted(away_prod)

    a = len(A)
    b = len(B)
    p = a + b
    c = M - p

    if p == 0:
        phi[M] = leaf_val
        return phi

    P0 = 1.0
    for fid, o in toward:
        P0 *= o
    for fid, z in away:
        P0 *= z

    toward_o = [t[1] for t in A]

    a_subsets = 1 << a
    S_A = [0.0] * (a + 1)
    for mask in range(a_subsets):
        k = mask.bit_count()
        inv_prod_o = 1.0
        for i in range(a):
            if mask & (1 << i):
                inv_prod_o /= toward_o[i]
        S_A[k] += inv_prod_o

    W = []
    for k in range(a + 1):
        total = 0.0
        for t in range(c + 1):
            total += _binomial(c, t) * _weight(k + t, M)
        W.append(total)

    for j_idx, (fid, o_j) in enumerate(A):
        z_j = 1.0 - o_j
        S_A_without = [0.0] * a
        for mask in range(a_subsets):
            if mask & (1 << j_idx):
                continue
            k = mask.bit_count()
            inv_prod_o = 1.0
            for i in range(a):
                if i == j_idx:
                    continue
                if mask & (1 << i):
                    inv_prod_o /= toward_o[i]
            S_A_without[k] += inv_prod_o
        total = 0.0
        for k in range(a):
            total += S_A_without[k] * W[k]
        phi[fid] += leaf_val * z_j / o_j * P0 * total

    total_away = 0.0
    for k in range(a + 1):
        total_away += S_A[k] * W[k]
    contribution_away = -leaf_val * P0 * total_away
    for fid in B:
        phi[fid] += contribution_away

    phi[M] += leaf_val * P0
    return phi


def _tree_shap(tree, x, M, val_fn=None):
    """TreeSHAP for a single tree and a single sample.

    Parameters
    ----------
    tree : _Tree
    x : list of float
    M : int
        Total number of features.
    val_fn : callable or None
        Extracts the scalar leaf value from a node.
        If None, uses ``node.prediction`` directly.

    Returns
    -------
    phi : list of length M+1, where phi[M] is the bias.
    """
    if val_fn is None:

        def val_fn(node):
            return node.prediction

    phi = [0.0] * (M + 1)

    def _recurse(node, toward, away):
        if node.is_leaf:
            contrib = _leaf_contributions(val_fn(node), toward, away, M)
            for i in range(M + 1):
                phi[i] += contrib[i]
            return

        f = node.feature
        left_w = node.left.size / node.size
        right_w = node.right.size / node.size

        if x[f] <= node.threshold:
            toward.append((f, left_w))
            _recurse(node.left, toward, away)
            toward.pop()
            away.append((f, right_w))
            _recurse(node.right, toward, away)
            away.pop()
        else:
            toward.append((f, right_w))
            _recurse(node.right, toward, away)
            toward.pop()
            away.append((f, left_w))
            _recurse(node.left, toward, away)
            away.pop()

    _recurse(tree.root, [], [])
    return phi


# ---------------------------------------------------------------------------
# Forest-level SHAP
# ---------------------------------------------------------------------------


def forest_shap_values(forest, X):
    """Compute SHAP values for a fitted RandomForest.

    Returns (shap_values, bias).

    Regressor:
        shap_values shape (n_samples, n_features), bias is float.
    Classifier:
        shap_values shape (n_samples, n_features, n_classes), bias is list
        of length n_classes.
    """
    M = forest.n_features_in_
    n_trees = len(forest.trees_)
    if n_trees == 0:
        raise ValueError("forest has no fitted trees")

    X_list = _to_list(X)
    n = len(X_list)
    is_clf = forest.task == "classifier"

    if is_clf:
        C = forest.n_classes_
        shap = [[[0.0] * C for _ in range(M)] for _ in range(n)]
        bias = [0.0] * C

        def _leaf_proba(node, c):
            p = node.prediction
            if p is None:
                return 0.0
            if isinstance(p, list):
                total = sum(p)
                if total == 0:
                    return 0.0
                return float(p[c]) / total
            return float(p)

        for tree in forest.trees_:
            # Pre-compute class-specific bias for this tree (same for all x)
            tree_bias = []
            for c in range(C):
                phi_c = _tree_shap(tree, X_list[0], M, val_fn=lambda node, cc=c: _leaf_proba(node, cc))
                tree_bias.append(phi_c[M])

            for c in range(C):
                bias[c] += tree_bias[c]
            for si in range(n):
                for c in range(C):

                    def _make_vfn(cc):
                        return lambda node: _leaf_proba(node, cc)

                    phi = _tree_shap(tree, X_list[si], M, val_fn=_make_vfn(c))
                    for fi in range(M):
                        shap[si][fi][c] += phi[fi]

        inv_n = 1.0 / n_trees
        for si in range(n):
            for fi in range(M):
                for c in range(C):
                    shap[si][fi][c] *= inv_n
        for c in range(C):
            bias[c] *= inv_n
        return shap, bias
    else:
        shap = [[0.0] * M for _ in range(n)]
        bias = 0.0
        for tree in forest.trees_:
            phi_ref = _tree_shap(tree, X_list[0], M)
            bias += phi_ref[M]
            for si in range(n):
                phi = _tree_shap(tree, X_list[si], M)
                for fi in range(M):
                    shap[si][fi] += phi[fi]
        inv_n = 1.0 / n_trees
        for si in range(n):
            for fi in range(M):
                shap[si][fi] *= inv_n
        bias *= inv_n
        return shap, bias
