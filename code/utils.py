import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_PLACES = [HERE, os.path.join(HERE, "data"),
           os.path.join(HERE, "..", "Final_Project_CMPSCI_589_Fall2026_Supporting_Files")]
# label column + which label is the positive class for F1 (None = more than 2 classes, labels 0..c-1)
LABELS = {"parkinsons.csv": ("Diagnosis", "1"), "rice.csv": ("label", "Cammeo"), "credit_approval.csv": ("label", "1"),
          "cmc.csv": ("label", None)}

def load(name):
    # returns X (categories numbered like hw3), y, kinds ("num"/"cat"), n_classes
    if name == "digits":
        from sklearn import datasets
        X, y = datasets.load_digits(return_X_y=True)
        return X.astype(float), y.astype(int), ["num"]*X.shape[1], 10
    path = next((os.path.join(d, name) for d in _PLACES if os.path.isfile(os.path.join(d, name))), name)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    col, pos = LABELS[name]
    target = header.index(col)
    if pos is None:
        y = np.array([int(r[target]) for r in rows])
    else:
        y = np.array([1 if r[target] == pos else 0 for r in rows])
    kinds, cols = [], []
    for i in range(len(header)):
        if i == target:
            continue
        raw = [r[i] for r in rows]
        if header[i].endswith("_cat"):
            code = {v: j for j, v in enumerate(sorted(set(raw)))}
            kinds.append("cat")
            cols.append(np.array([code[v] for v in raw], dtype=float))
        else:
            kinds.append("num")   # parkinsons has no _num suffix, all numerical
            cols.append(np.array(raw, dtype=float))
    return np.column_stack(cols), y, kinds, 2 if pos is not None else len(np.unique(y))

def one_hot(X, kinds):
    # knn and nn need numbers: one 0/1 column per category value
    cols = []
    for i, k in enumerate(kinds):
        if k == "cat":
            vals = np.unique(X[:, i])
            cols.append((X[:, i][:, None] == vals[None, :]).astype(float))
        else:
            cols.append(X[:, i][:, None])
    return np.hstack(cols)
def normalizer(X):
    # min/max from train only
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi-lo, 1.0)
    return lambda Z: (Z-lo)/span

def stratified_folds(y, k, rng):
    # shuffle each class separately, then deal round-robin into the k folds
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]
def f1_of(y_true, y_pred, positive):
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    precision = tp/(tp + fp) if tp + fp else 0.0
    recall = tp/(tp + fn) if tp + fn else 0.0
    return 2*precision*recall/(precision + recall) if precision + recall else 0.0
def metrics(y_true, y_pred, n_classes):
    acc = float(np.mean(y_true == y_pred))
    if n_classes == 2:
        return {"accuracy": acc, "f1": f1_of(y_true, y_pred, 1)}
    # more than 2 classes: macro F1 = mean of each class's F1 (that class as positive)
    return {"accuracy": acc, "f1": float(np.mean([f1_of(y_true, y_pred, c) for c in range(n_classes)]))}
