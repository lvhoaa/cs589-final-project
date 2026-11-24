import json
import math
import os
import sys
import time
import numpy as np
from decision_tree import majority_class
from extra_trees import train_extra_trees
from knn import KNNClassifier
from neural_net import cost, predict, train
from plotting import knn_curve, learning_curve, ntree_curve, rf_vs_et
from random_forest import majority_vote, train_forest, tree_votes
from utils import load, metrics, normalizer, one_hot, stratified_folds

SEED = 589
K = 10
DATASETS = [("digits", "Digits"), ("parkinsons.csv", "Parkinsons"), ("rice.csv", "Rice"), ("credit_approval.csv", "Credit"),
            ("cmc.csv", "CMC")]   # cmc = extra credit 2
KS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
# (minimum_size_for_split, minimum_gain, maximum_depth)
STOP_GRID = [(2, 0.0, None), (5, 0.0, None), (10, 0.0, None), (20, 0.0, None), (5, 0.0, 5), (5, 0.0, 10)]
TUNE_NTREE = 10
NTREES = [1, 5, 10, 20, 30, 40, 50]
BATCH = 32
ALPHAS = [0.3, 1, 3]
EPOCHS = [100, 300]
TUNE_ARCH = [16]
ARCHS = [[8], [16], [32], [64], [16, 16], [32, 32]]
LAMBDAS = [0, 0.1, 1]
CURVE_NS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

def splits(y, rng):
    folds = stratified_folds(y, K, rng)
    for j in range(K):
        yield np.concatenate([folds[i] for i in range(K) if i != j]), folds[j]
def summarize(per_fold):
    out = {m: float(np.mean([r[m] for r in per_fold])) for m in ("accuracy", "f1")}
    out.update({m + "_std": float(np.std([r[m] for r in per_fold])) for m in ("accuracy", "f1")})
    return out
def targets(y, nc):
    # one output neuron for 2 classes, one per class otherwise
    return np.eye(nc)[y] if nc > 2 else y[:, None].astype(float)
def knn_cv(Xo, y, nc):
    per_k = {k: [] for k in KS}
    for tr, te in splits(y, np.random.default_rng(SEED)):
        norm = normalizer(Xo[tr])
        model = KNNClassifier(1).fit(norm(Xo[tr]), y[tr])
        for k in KS:
            model.k = k
            per_k[k].append(metrics(y[te], model.predict(norm(Xo[te])), nc))
    return {k: summarize(per_k[k]) for k in KS}
def rf_cv(X, y, kinds, nc, stop, ntrees, trainer=train_forest, times=None):
    rng = np.random.default_rng(SEED)
    per_n ={n: [] for n in ntrees}
    for tr, te in splits(y, rng):
        # train the biggest forest once, smaller ones are its first n trees (same as hw3)
        t0=time.time()
        trees = trainer(X[tr], y[tr], kinds, max(ntrees), rng, stop)
        if times is not None:
            times.append(time.time() - t0)
        votes = tree_votes(trees, X[te])
        for n in ntrees:
            per_n[n].append(metrics(y[te], majority_vote(votes[:, :n]), nc))
    return {n: summarize(per_n[n]) for n in ntrees}
def nn_cv(Xo, y, nc, arch, lam, alpha, epochs):
    rng = np.random.default_rng(SEED)
    per_fold = []
    for tr, te in splits(y, rng):
        norm = normalizer(Xo[tr])
        Y = targets(y[tr], nc)
        thetas,_ = train(norm(Xo[tr]), Y, [Xo.shape[1]] + arch + [Y.shape[1]], lam, alpha, epochs, BATCH, rng)
        per_fold.append(metrics(y[te], predict(thetas, norm(Xo[te])), nc))
    return summarize(per_fold)
def nn_curve(Xo, y, nc, arch, lam, alpha, epochs):
    # hw4 learning curve: train from scratch on first n instances, J on test fold
    rng = np.random.default_rng(SEED)
    fs = list(splits(y, rng))
    smallest = min(len(t) for t,_ in fs)
    ns = [n for n in CURVE_NS if n < smallest] + [smallest]
    steps = epochs*math.ceil(smallest/BATCH)
    J = []
    for tr, te in fs:
        norm = normalizer(Xo[tr])
        # stratified chunks (~5, or ~20 for 10 classes) so each prefix keeps the class ratio
        size = 5 if nc == 2 else 20
        order = np.concatenate([tr[c] for c in stratified_folds(y[tr], len(tr)//size, rng)])
        row = []
        for n in ns:
            rows = order[:n]
            Y = targets(y[rows], nc)
            thetas,_ = train(norm(Xo[rows]), Y, [Xo.shape[1]] + arch + [Y.shape[1]], lam, alpha,
                             math.ceil(steps/math.ceil(n/BATCH)), BATCH, rng)
            row.append(cost(thetas, norm(Xo[te]), targets(y[te], nc), 0))
        J.append(row)
    J = np.array(J)
    return ns, J.mean(axis=0).tolist(), J.std(axis=0).tolist()
def ensemble_cv(X, Xo, y, kinds, nc, members):
    # extra credit 3: every member gets its own bootstrap of the training folds, then majority vote
    rng = np.random.default_rng(SEED)
    per = {m["name"]: [] for m in members}
    per["ensemble"] = []
    for tr, te in splits(y, rng):
        preds = []
        for m in members:
            b = tr[rng.integers(0, len(tr), size=len(tr))]
            if m["algo"] == "rf":
                p = majority_vote(tree_votes(train_forest(X[b], y[b], kinds, m["ntree"], rng, m["stop"]), X[te]))
            else:
                norm = normalizer(Xo[b])
                if m["algo"] == "knn":
                    p = KNNClassifier(m["k"]).fit(norm(Xo[b]), y[b]).predict(norm(Xo[te]))
                else:
                    Y = targets(y[b], nc)
                    thetas,_ = train(norm(Xo[b]), Y, [Xo.shape[1]] + m["arch"] + [Y.shape[1]], m["lam"],
                                     m["alpha"], m["epochs"], BATCH, rng)
                    p = predict(thetas, norm(Xo[te]))
            preds.append(p)
            per[m["name"]].append(metrics(y[te], p, nc))
        votes = np.column_stack(preds)
        per["ensemble"].append(metrics(y[te], np.array([majority_class(r) for r in votes]), nc))
    return {k: summarize(v) for k, v in per.items()}
def name(arch):
    return "-".join(str(a) for a in arch)
def stop_name(s):
    return f"{s[0]}|{s[1]}|{s[2]}"

def main():
    os.makedirs(OUT, exist_ok=True)
    chosen = [d for d in DATASETS if d[1].lower() in [a.lower() for a in sys.argv[1:]]] or DATASETS
    for filename, title in chosen:
        X, y, kinds, nc = load(filename)
        Xo = one_hot(X, kinds)
        t = title.lower()
        print(f"== {title}: {X.shape[0]} instances, {X.shape[1]} attributes, {Xo.shape[1]} inputs after one-hot", flush=True)
        res = {"seed": SEED, "k_folds": K, "n_instances": len(y), "n_attributes": X.shape[1], "n_inputs": Xo.shape[1],
               "n_categorical": kinds.count("cat"), "n_classes": nc, "class_counts": np.bincount(y).tolist()}

        knn = knn_cv(Xo, y, nc)
        best_k = max(KS, key=lambda k: (knn[k]["f1"], knn[k]["accuracy"]))
        for k in KS:
            print(f"   knn k={k:<3} acc={knn[k]['accuracy']:.4f} f1={knn[k]['f1']:.4f}", flush=True)
        knn_curve(KS, [knn[k]["accuracy"] for k in KS], [knn[k]["f1"] for k in KS], f"{title}: $k$-NN", f"{t}_knn", OUT)
        res["knn"] = {"results": {str(k): knn[k] for k in KS}, "best_k": best_k}

        tuning = []
        for s in STOP_GRID:
            stop = {"min_size": s[0], "min_gain": s[1], "max_depth": s[2]}
            r = rf_cv(X, y, kinds, nc, stop, [TUNE_NTREE])[TUNE_NTREE]
            tuning.append({"stop": stop, **r})
            print(f"   rf stop={stop_name(s):<12} acc={r['accuracy']:.4f} f1={r['f1']:.4f}", flush=True)
        stop = max(tuning, key=lambda r: (r["f1"], r["accuracy"]))["stop"]
        rf = rf_cv(X, y, kinds, nc, stop, NTREES)
        best_ntree = max(NTREES, key=lambda n: (rf[n]["f1"], rf[n]["accuracy"]))
        for n in NTREES:
            print(f"   rf ntree={n:<3} acc={rf[n]['accuracy']:.4f} f1={rf[n]['f1']:.4f}", flush=True)
        ntree_curve(NTREES, [rf[n]["accuracy"] for n in NTREES], [rf[n]["f1"] for n in NTREES],
                    f"{title}: random forest", f"{t}_rf", OUT)
        res["rf"] = {"tuning": tuning, "stop": stop, "results": {str(n): rf[n] for n in NTREES}, "best_ntree": best_ntree}

        nn_tuning = []
        for alpha in ALPHAS:
            for epochs in EPOCHS:
                r = nn_cv(Xo, y, nc, TUNE_ARCH, 0, alpha, epochs)
                nn_tuning.append({"alpha": alpha, "epochs": epochs, **r})
                print(f"   nn tune alpha={alpha:<4} m={epochs:<4} acc={r['accuracy']:.4f} f1={r['f1']:.4f}", flush=True)
        best = max(nn_tuning, key=lambda r: (r["f1"], r["accuracy"]))
        alpha, epochs = best["alpha"], best["epochs"]
        grid = {}
        for arch in ARCHS:
            for lam in LAMBDAS:
                r = nn_cv(Xo, y, nc, arch, lam, alpha, epochs)
                grid[f"{name(arch)}|{lam}"] = r
                print(f"   nn {name(arch):>6} lam={lam:<4} acc={r['accuracy']:.4f} f1={r['f1']:.4f}", flush=True)
        ranked = sorted(grid, key=lambda k: (grid[k]["f1"], grid[k]["accuracy"]), reverse=True)
        best_arch = [int(a) for a in ranked[0].split("|")[0].split("-")]
        best_lam = float(ranked[0].split("|")[1])
        ns, mean, std = nn_curve(Xo, y, nc, best_arch, best_lam, alpha, epochs)
        learning_curve(ns, mean, std, f"{title}: [{Xo.shape[1]} {' '.join(map(str, best_arch))} {targets(y[:1], nc).shape[1]}], "
                       + r"$\alpha$" + f"={alpha:g}, " + r"$\lambda$" + f"={best_lam:g}", f"{t}_nn_curve", OUT)
        res["nn"] = {"tuning": nn_tuning, "alpha": alpha, "epochs": epochs, "grid": grid,
                     "best_arch": best_arch, "best_lambda": best_lam, "curve": {"n": ns, "J_mean": mean, "J_std": std}}

        # ensemble: best knn, best forest, 3 best nets with different architectures
        nets, seen = [], set()
        for key in ranked:
            a, l = key.split("|")
            if a not in seen:
                seen.add(a)
                nets.append({"name": f"nn {a} lam={l}", "algo": "nn", "arch": [int(v) for v in a.split("-")], "lam": float(l),
                             "alpha": alpha, "epochs": epochs})
            if len(nets) == 3:
                break
        members = [{"name": f"knn k={best_k}", "algo": "knn", "k": best_k},
                   {"name": f"rf ntree={best_ntree}", "algo": "rf", "ntree": best_ntree, "stop": stop}] + nets
        ens = ensemble_cv(X, Xo, y, kinds, nc, members)
        for k, v in ens.items():
            print(f"   ens {k:<22} acc={v['accuracy']:.4f} f1={v['f1']:.4f}", flush=True)
        res["ensemble"] = {"members": [m["name"] for m in members], "results": ens}

        # extra credit 4: extra trees, same tuning as the forest, then compare with it
        et_tuning = []
        for s in STOP_GRID:
            st = {"min_size": s[0], "min_gain": s[1], "max_depth": s[2]}
            r = rf_cv(X, y, kinds, nc, st, [TUNE_NTREE], train_extra_trees)[TUNE_NTREE]
            et_tuning.append({"stop": st, **r})
            print(f"   et stop={stop_name(s):<12} acc={r['accuracy']:.4f} f1={r['f1']:.4f}", flush=True)
        et_stop = max(et_tuning, key=lambda r: (r["f1"], r["accuracy"]))["stop"]
        et_times, rf_times = [], []
        et = rf_cv(X, y, kinds, nc, et_stop, NTREES, train_extra_trees, et_times)
        rf_cv(X, y, kinds, nc, stop, [max(NTREES)], train_forest, rf_times)   # only to time the forest the same way
        best_et = max(NTREES, key=lambda n: (et[n]["f1"], et[n]["accuracy"]))
        for n in NTREES:
            print(f"   et ntree={n:<3} acc={et[n]['accuracy']:.4f} f1={et[n]['f1']:.4f}", flush=True)
        rf_vs_et(NTREES, [rf[n]["accuracy"] for n in NTREES], [et[n]["accuracy"] for n in NTREES],
                 [rf[n]["f1"] for n in NTREES], [et[n]["f1"] for n in NTREES], f"{title}: random forest vs. extra trees", f"{t}_et", OUT)
        res["et"] = {"tuning": et_tuning, "stop": et_stop, "results": {str(n): et[n] for n in NTREES}, "best_ntree": best_et,
                     "train_seconds_50": float(np.mean(et_times)), "rf_train_seconds_50": float(np.mean(rf_times))}

        with open(os.path.join(OUT, f"results_{t}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
