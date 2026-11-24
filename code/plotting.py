import matplotlib
matplotlib.use("Agg")   # no window, only save files
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#1f4e9e"
RED = "#c0392b"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, stem, out_dir):
    fig.savefig(f"{out_dir}/{stem}.pdf")
    fig.savefig(f"{out_dir}/{stem}.png")
    plt.close(fig)
def two_metrics(ax, xs, acc, f1):
    ax.plot(xs, acc, color=BLUE, linewidth=1.6, marker="D", markersize=5, zorder=3, label="Accuracy")
    ax.plot(xs, f1, color=RED, linewidth=1.6, marker="o", markersize=5, zorder=3, label="F1 score")
    ax.legend(loc="best", fontsize=9, framealpha=0.95, edgecolor="#bbbbbb")
def knn_curve(ks, acc, f1, title, stem, out_dir):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    two_metrics(ax, ks, acc, f1)
    ax.set_xlabel("Value of $k$")
    ax.set_ylabel("10-fold CV performance")
    ax.set_title(title)
    ax.set_xticks([k for k in ks if k not in (3, 7, 9)])
    save(fig, stem, out_dir)
def ntree_curve(ntrees, acc, f1, title, stem, out_dir):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    two_metrics(ax, ntrees, acc, f1)
    ax.set_xlabel("Number of trees ($ntree$)")
    ax.set_ylabel("10-fold CV performance")
    ax.set_title(title)
    ax.set_xticks(ntrees)
    save(fig, stem, out_dir)
def learning_curve(ns, mean, std, title, stem, out_dir):
    mean, std = np.array(mean), np.array(std)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.fill_between(ns, mean - std, mean + std, color=BLUE, alpha=0.15, linewidth=0, label="$\\pm$1 std over folds")
    ax.plot(ns, mean, color=BLUE, linewidth=1.6, marker="o", markersize=3.5, zorder=3, label="test $J$ (mean of 10 folds)")
    ax.set_xlabel("Number of training instances shown to the network")
    ax.set_ylabel("Cost $J$ on the test fold")
    ax.set_title(title)
    ax.set_xlim(0, ns[-1]+5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, edgecolor="#bbbbbb")
    save(fig, stem, out_dir)
def rf_vs_et(ntrees, rf_acc, et_acc, rf_f1, et_f1, title, stem, out_dir):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(ntrees, rf_acc, color=BLUE, linewidth=1.6, marker="D", markersize=5, label="RF accuracy")
    ax.plot(ntrees, et_acc, color=BLUE, linewidth=1.6, marker="D", markersize=5, linestyle="--", label="ET accuracy")
    ax.plot(ntrees, rf_f1, color=RED, linewidth=1.6, marker="o", markersize=5, label="RF F1")
    ax.plot(ntrees, et_f1, color=RED, linewidth=1.6, marker="o", markersize=5, linestyle="--", label="ET F1")
    ax.set_xlabel("Number of trees ($ntree$)")
    ax.set_ylabel("10-fold CV performance")
    ax.set_title(title)
    ax.set_xticks(ntrees)
    ax.legend(loc="best", fontsize=8, framealpha=0.95, edgecolor="#bbbbbb", ncol=2)
    save(fig, stem, out_dir)
