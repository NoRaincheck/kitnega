import hashlib
import random


class Dice:
    def __init__(self, seed=None, bias=None, filtered=False):
        if seed is not None:
            if isinstance(seed, str):
                seed = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (1 << 32)
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()
        self._bias = bias
        self._filtered = filtered

    def roll(self, expr):
        expr = expr.lower().replace(" ", "")
        mod = 0
        if "+" in expr:
            expr, mod = expr.split("+", 1)
            mod = int(mod)
        elif "-" in expr and expr.index("-") > (expr.index("d") if "d" in expr else -1):
            expr, mod = expr.rsplit("-", 1)
            mod = -int(mod)
        if "d" in expr:
            parts = expr.split("d")
            num = int(parts[0]) if parts[0] else 1
            sides = int(parts[1])
            return sum(self._rng.randint(1, sides) for _ in range(num)) + mod
        return int(expr) + mod

    def pick(self, items, weights=None):
        if not items:
            return None
        if weights:
            return self._rng.choices(items, weights=weights, k=1)[0]
        return self._rng.choice(items)

    def picks(self, items, count, weights=None):
        if not items:
            return []
        if weights:
            return list(self._rng.choices(items, weights=weights, k=count))
        return [self._rng.choice(items) for _ in range(count)]

    def sample(self, items, count):
        k = min(count, len(items))
        return self._rng.sample(items, k)

    def shuffle(self, items):
        items = list(items)
        self._rng.shuffle(items)
        return items

    def weighted(self, table, bias=None):
        items = list(table.items()) if isinstance(table, dict) else table
        bias_fn = bias or self._bias
        if bias_fn:
            if self._filtered:
                filtered_items = [(item, w) for item, w in items if bias_fn(item)]
                if filtered_items:
                    items = filtered_items
            else:
                total = sum(w for _, w in items)
                items = [(item, w + total if bias_fn(item) else w) for item, w in items]
        total = sum(w for _, w in items)
        r = self._rng.random() * total
        for item, weight in items:
            r -= weight
            if r <= 0:
                return item
        return items[-1][0]
