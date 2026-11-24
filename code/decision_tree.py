from collections import Counter
import numpy as np

# same nested dicts as hw1, plus a branch kind for numerical attributes
# leaf->{"label": c}
# decision node->{"attribute": a, "children": {value: subtree}, "majority": c}
# numeric node->{"attribute": a, "threshold": t, "low": subtree, "high": subtree, "majority": c}

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))
def majority_class(y):
    # most common class
    return Counter(y.tolist()).most_common(1)[0][0]
def split_score(column, y, criterion):
    impurity = entropy if criterion == "info_gain" else gini
    total = 0.0
    for v in np.unique(column):
        part = y[column == v]
        total += len(part) / len(y) * impurity(part)
    return total

def numeric_split_score(column, y, criterion):
    # same weighted impurity, 
    # two branches, cut at mean
    impurity = entropy if criterion == "info_gain" else gini
    threshold = float(column.mean())
    low = column <= threshold
    if low.all() or not low.any():
        return np.inf, None
    total = (low.sum() / len(y) * impurity(y[low])+ (~low).sum() / len(y) * impurity(y[~low]))
    return total, threshold

def build_tree(X, y, attributes, kinds, m, rng, criterion="info_gain",
               min_size=2, min_gain=0.0, max_depth=None, depth=0):
    maj = majority_class(y)

    if len(np.unique(y)) == 1:
        return {"label": y[0]}
    if not attributes:
        return {"label": maj}
    if len(y) < min_size or (max_depth is not None and depth >= max_depth):
        return {"label": maj}

    # random forest part: only m are candidates at this node
    pool = attributes if len(attributes) <= m else rng.choice(attributes, m, replace=False)
    scored = []
    for a in pool:
        if kinds[a] == "cat":
            scored.append((split_score(X[:, a], y, criterion), int(a), None))
        else:
            score, threshold = numeric_split_score(X[:, a], y, criterion)
            scored.append((score, int(a), threshold))

    score, best, threshold = min(scored)
    if (entropy if criterion == "info_gain" else gini)(y) - score <= min_gain:
        return {"label": maj}

    if kinds[best] == "cat":
        rest = [a for a in attributes if a != best]
        children = {}
        for v in np.unique(X[:, best]):
            rows = X[:, best] == v
            children[v] = build_tree(X[rows], y[rows], rest, kinds, m, rng, criterion,min_size, min_gain, max_depth, depth + 1)
        return {"attribute": best, "children": children, "majority": maj}

    # a numerical attribute stays in the list, a node further down can cut it again
    low = X[:, best] <= threshold
    return {"attribute": best, 
            "threshold": threshold, 
            "majority": maj,
            "low": build_tree(X[low], y[low], attributes, kinds, m, rng, criterion,
                              min_size, min_gain, max_depth, depth + 1),
            "high": build_tree(X[~low], y[~low], attributes, kinds, m, rng, criterion,
                               min_size, min_gain, max_depth, depth + 1)}

def predict_one(tree, x):
    node = tree
    while "label" not in node:
        if "threshold" in node:
            node = node["low"] if x[node["attribute"]] <= node["threshold"] else node["high"]
            continue
        child = node["children"].get(x[node["attribute"]])
        if child is None: # unseen?-> fall back to majority
            return node["majority"]
        node = child
    return node["label"]

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])
