"""A multilayer perceptron (MLP) with manually written backpropagation.

Architecture
------------
input -> [Linear -> activation] x (L-1 hidden layers) -> Linear -> softmax

Loss: cross-entropy. Optimizer: mini-batch stochastic gradient descent.

Forward pass (layer i):    z_i = a_{i-1} @ W_i + b_i ,   a_i = f(z_i)
Backward pass:
    output layer (softmax + cross-entropy) :  delta_L = (probs - y) / m
    gradients                              :  dW_i = a_{i-1}.T @ delta_i
                                              db_i = sum over batch of delta_i
    propagate to previous layer            :  delta_{i-1} = (delta_i @ W_i.T) * f'(a_{i-1})
"""
import numpy as np

from .activations import ACTIVATIONS, softmax
from .losses import cross_entropy, one_hot


class MLP:
    def __init__(self, layer_sizes, activation="tanh", seed=0):
        """layer_sizes example: [64, 32, 10] = 64 inputs, one hidden layer of 32, 10 classes."""
        self.layer_sizes = layer_sizes
        self.act, self.act_grad = ACTIVATIONS[activation]
        rng = np.random.default_rng(seed)

        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))  # Xavier/Glorot uniform scale
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        self.n_layers = len(self.W)
        self._cache = None

    # ---------------------------------------------------------------- forward
    def forward(self, X):
        """Return class probabilities, shape (m, n_classes). Stores activations for backward()."""
        activations = [X]
        a = X
        for i in range(self.n_layers - 1):
            a = self.act(a @ self.W[i] + self.b[i])
            activations.append(a)
        probs = softmax(a @ self.W[-1] + self.b[-1])
        activations.append(probs)
        self._cache = activations
        return probs

    # --------------------------------------------------------------- backward
    def backward(self, y_onehot):
        """Backpropagation. Must be called right after forward() on the same batch.

        Returns lists of gradients (dW, db), one entry per layer.
        """
        acts = self._cache
        m = y_onehot.shape[0]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        delta = (acts[-1] - y_onehot) / m  # gradient of mean loss w.r.t. output logits
        for i in reversed(range(self.n_layers)):
            dW[i] = acts[i].T @ delta
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.act_grad(acts[i])
        return dW, db

    # --------------------------------------------------------------- training
    def loss(self, X, y):
        return cross_entropy(self.forward(X), one_hot(y, self.layer_sizes[-1]))

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y, X_val=None, y_val=None, epochs=50, lr=0.1, batch_size=32,
            seed=0, verbose=True):
        """Train with mini-batch SGD. Returns a dict with loss/accuracy history."""
        rng = np.random.default_rng(seed)
        n_classes = self.layer_sizes[-1]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            order = rng.permutation(X.shape[0])  # shuffle every epoch
            for start in range(0, X.shape[0], batch_size):
                idx = order[start:start + batch_size]
                self.forward(X[idx])
                dW, db = self.backward(one_hot(y[idx], n_classes))
                for i in range(self.n_layers):  # SGD update
                    self.W[i] -= lr * dW[i]
                    self.b[i] -= lr * db[i]

            history["train_loss"].append(self.loss(X, y))
            history["train_acc"].append(self.accuracy(X, y))
            msg = f"epoch {epoch:3d}  loss {history['train_loss'][-1]:.4f}  acc {history['train_acc'][-1]:.4f}"
            if X_val is not None:
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_acc"].append(self.accuracy(X_val, y_val))
                msg += f"  val_loss {history['val_loss'][-1]:.4f}  val_acc {history['val_acc'][-1]:.4f}"
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                print(msg)
        return history
