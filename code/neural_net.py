import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-np.clip(z, -500, 500)))
def init_weights(sizes, rng):
    # row i = neuron i, col 0 = bias
    return [rng.uniform(-1, 1, size=(sizes[l+1], sizes[l]+1)) for l in range(len(sizes)-1)]
def add_bias(a):
    return np.hstack([np.ones((a.shape[0], 1)), a])

def forward(thetas, X):
    a= add_bias(X)
    acts, zs = [a], []
    for l, theta in enumerate(thetas):
        z = a @ theta.T
        a = sigmoid(z)
        if l < len(thetas)-1:
            a = add_bias(a)
        zs.append(z)
        acts.append(a)
    return acts, zs
def instance_costs(f, Y):
    f= np.clip(f, 1e-12, 1-1e-12)
    return np.sum(-Y*np.log(f) - (1-Y)*np.log(1-f), axis=1)
def penalty(thetas, lam, n):
    return lam/(2*n) * sum(float(np.sum(t[:, 1:]**2)) for t in thetas)
def cost(thetas, X, Y, lam):
    f = forward(thetas, X)[0][-1]
    return float(np.mean(instance_costs(f, Y))) + penalty(thetas, lam, len(X))

def deltas(thetas, acts, Y):
    ds=[acts[-1] - Y]
    for l in range(len(thetas)-1, 0, -1):
        a= acts[l]
        ds.insert(0, (ds[0] @ thetas[l])[:, 1:] * a[:, 1:] * (1-a[:, 1:]))
    return ds
def gradients(thetas, X, Y, lam, n_total=None):
    n = len(X)
    n_total = n_total or n   # minibatch: reg term still uses full train size
    acts,_ = forward(thetas, X)
    ds = deltas(thetas, acts, Y)
    grads=[]
    for l, theta in enumerate(thetas):
        p = lam/n_total*theta
        p[:, 0] = 0
        grads.append(ds[l].T @ acts[l]/n + p)
    return grads

def train(X, Y, sizes, lam, alpha, epochs, batch, rng, track=None):
    thetas = init_weights(sizes, rng)
    hist = []
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            rows=order[s:s+batch]
            for theta, g in zip(thetas, gradients(thetas, X[rows], Y[rows], lam, len(X))):
                theta -= alpha*g
        if track is not None:
            hist.append(cost(thetas, track[0], track[1], lam))
    return thetas, hist
def predict(thetas, X):
    f = forward(thetas, X)[0][-1]
    if f.shape[1] > 1:   # digits: one output per class
        return np.argmax(f, axis=1)
    return (f[:, 0] >= 0.5).astype(int)
