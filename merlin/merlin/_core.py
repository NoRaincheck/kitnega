from lib.array import ndarray


class _Node:
    __slots__ = (
        "feature", "threshold", "left", "right", "is_leaf",
        "prediction", "size", "depth",
        "tau", "delta", "lower_bounds", "upper_bounds",
    )

    def __init__(self, tau=0.0):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.prediction = None
        self.size = 0
        self.depth = 0
        self.tau = tau
        self.delta = 0.0
        self.lower_bounds = None
        self.upper_bounds = None

    def update_bounds(self, X):
        nf = len(X[0])
        if self.lower_bounds is None:
            lo = [float("inf")] * nf
            hi = [float("-inf")] * nf
            for x in X:
                for j in range(nf):
                    if x[j] < lo[j]:
                        lo[j] = x[j]
                    if x[j] > hi[j]:
                        hi[j] = x[j]
            self.lower_bounds = lo
            self.upper_bounds = hi
        else:
            for x in X:
                for j in range(nf):
                    if x[j] < self.lower_bounds[j]:
                        self.lower_bounds[j] = x[j]
                    if x[j] > self.upper_bounds[j]:
                        self.upper_bounds[j] = x[j]


def _to_list(x):
    if isinstance(x, ndarray):
        return [list(r.flat) for r in x]
    return list(x)


def _to_flat_list(y):
    if isinstance(y, ndarray):
        return list(y.flat)
    return list(y)


def _traverse(node, x):
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node


def _copy_node(node):
    if node is None:
        return None
    n = _Node(tau=node.tau)
    n.feature = node.feature
    n.threshold = node.threshold
    n.left = _copy_node(node.left)
    n.right = _copy_node(node.right)
    n.is_leaf = node.is_leaf
    n.prediction = node.prediction
    n.size = node.size
    n.depth = node.depth
    n.delta = node.delta
    n.lower_bounds = node.lower_bounds
    n.upper_bounds = node.upper_bounds
    return n
