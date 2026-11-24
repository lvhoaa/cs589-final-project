import numpy as np
from decision_tree import entropy, majority_class, split_score

# EC4: extra trees (Geurts 2006). vs my hw3 forest: no bootstrap + random cut instead of mean
# same dicts as hw3 so predict / tree_votes still work
def random_cut(column, y, rng):
    lo, hi= column.min(), column.max()
    if lo == hi:
        return np.inf, None
    t = rng.uniform(lo, hi)
    low = column <= t
    total = low.sum()/len(y)*entropy(y[low]) + (~low).sum()/len(y)*entropy(y[~low])
    return total, t
def build_extra_tree(X, y, attributes, kinds, m, rng, min_size=2, min_gain=0.0, max_depth=None, depth=0):
    maj = majority_class(y)
    if len(np.unique(y)) == 1:
        return {"label": y[0]}
    if not attributes or len(y) < min_size or (max_depth is not None and depth >= max_depth):
        return {"label": maj}
    pool= attributes if len(attributes) <= m else rng.choice(attributes, m, replace=False)
    scored = []
    for a in pool:
        if kinds[a] == "cat":
            scored.append((split_score(X[:, a], y, "info_gain"), int(a), None))
        else:
            score, t = random_cut(X[:, a], y, rng)
            scored.append((score, int(a), t))
    score, best, t = min(scored)
    if entropy(y) - score <= min_gain:
        return {"label": maj}
    if kinds[best] == "cat":
        rest = [a for a in attributes if a != best]
        children = {}
        for v in np.unique(X[:, best]):
            rows = X[:, best] == v
            children[v] = build_extra_tree(X[rows], y[rows], rest, kinds, m, rng, min_size, min_gain, max_depth, depth+1)
        return {"attribute": best, "children": children, "majority": maj}
    low = X[:, best] <= t
    return {"attribute": best, "threshold": t, "majority": maj,
            "low": build_extra_tree(X[low], y[low], attributes, kinds, m, rng, min_size, min_gain, max_depth, depth+1),
            "high": build_extra_tree(X[~low], y[~low], attributes, kinds, m, rng, min_size, min_gain, max_depth, depth+1)}
def train_extra_trees(X, y, kinds, ntree, rng, stop):
    m = max(1, int(round(np.sqrt(X.shape[1]))))
    attributes = list(range(X.shape[1]))
    # no bootstrap here
    return [build_extra_tree(X, y, attributes, kinds, m, rng, **stop) for _ in range(ntree)]
