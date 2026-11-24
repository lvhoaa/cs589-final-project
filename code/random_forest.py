import numpy as np
from decision_tree import build_tree, majority_class, predict
def train_forest(X, y, kinds, ntree, rng, stop):
    m= max(1,int(round(np.sqrt(X.shape[1]))))
    attributes= list(range(X.shape[1]))
    trees = []
    for _ in range(ntree):
        rows=rng.integers(0, len(y), size=len(y))   # bootstrap: n draws with replacement
        trees.append(build_tree(X[rows], y[rows], attributes, kinds, m, rng, **stop))
    return trees
def tree_votes(trees, X):
    # 1 row per test instance
    # 1 col per tree
    return np.column_stack([predict(t, X) for t in trees] )
def majority_vote(votes):
    return np.array([majority_class(row) for row in votes]    )
