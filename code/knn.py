import numpy as np

class KNNClassifier:
    def __init__(self, k=1):
        if k < 1:
            print('not taking k <1')
            raise ValueError("k must >= 1")
        self.k = k
    def fit(self, X, y):
        # just store data for later infer
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self
    def _euclidean_distances(self, X):
        # 
        X = np.asarray(X, dtype=float)
        sq_x = np.sum(X ** 2, axis=1)[:, None]
        sq_t = np.sum(self.X_train ** 2, axis=1)[None, :]
        # euclid dist: (x-y)^2 = x^2 -2xy + y^2
        d2 = sq_x - 2.0 * X @ self.X_train.T + sq_t
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)
    def predict(self, X):
        # infer from saved table
        X = np.asarray(X, dtype=float)
        dist = self._euclidean_distances(X)
        k = min(self.k, self.X_train.shape[0])
        # sort to get k smallest dists => to get the pool
        k_smallest_distances_from_X = np.argsort(dist, axis=1)[:, :k]
        labels_of_smallest_distances_from_X = self.y_train[k_smallest_distances_from_X]  # shape (n_queries, k)

        # take majority from k neighbors
        preds = np.empty(X.shape[0], dtype=self.y_train.dtype)
        for i, labels in enumerate(labels_of_smallest_distances_from_X):
            values, counts = np.unique(labels, return_counts=True)
            # get max idx with argmax
            preds[i] = values[np.argmax(counts)]
        return preds
