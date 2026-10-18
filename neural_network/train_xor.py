"""Why we need hidden layers: a perceptron solves AND but cannot solve XOR; an MLP can.

Run:  python train_xor.py
Saves plots into results/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import MLP, Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1])
y_xor = np.array([0, 1, 1, 0])


def plot_boundary(predict_fn, title, path):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predict_fn(grid).reshape(xx.shape)
    plt.figure(figsize=(4, 4))
    plt.contourf(xx, yy, zz, alpha=0.3, levels=[-0.5, 0.5, 1.5])
    plt.scatter(X[:, 0], X[:, 1], c=y_xor if "XOR" in title else y_and, edgecolors="k", s=120)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# 1) Perceptron on AND (linearly separable) -> converges
p_and = Perceptron(2, lr=0.1)
mistakes = p_and.fit(X, y_and, epochs=50)
print(f"Perceptron on AND: accuracy {np.mean(p_and.predict(X) == y_and):.2f}, "
      f"converged after {len(mistakes)} epochs")
plot_boundary(p_and.predict, "Perceptron on AND", "results/perceptron_and.png")

# 2) Perceptron on XOR (not linearly separable) -> never converges
p_xor = Perceptron(2, lr=0.1)
mistakes = p_xor.fit(X, y_xor, epochs=50)
print(f"Perceptron on XOR: accuracy {np.mean(p_xor.predict(X) == y_xor):.2f}, "
      f"mistakes in last epoch = {mistakes[-1]} (never reaches 0)")
plot_boundary(p_xor.predict, "Perceptron on XOR (fails)", "results/perceptron_xor.png")

# 3) MLP with one hidden layer on XOR -> solves it
mlp = MLP([2, 4, 2], activation="tanh", seed=0)
history = mlp.fit(X, y_xor, epochs=2000, lr=0.5, batch_size=4, verbose=False)
print(f"MLP [2,4,2] on XOR: accuracy {mlp.accuracy(X, y_xor):.2f}, final loss {history['train_loss'][-1]:.4f}")
plot_boundary(mlp.predict, "MLP on XOR (solves it)", "results/mlp_xor.png")

plt.figure(figsize=(5, 3.5))
plt.plot(history["train_loss"])
plt.xlabel("epoch")
plt.ylabel("cross-entropy loss")
plt.title("MLP training loss on XOR")
plt.tight_layout()
plt.savefig("results/mlp_xor_loss.png", dpi=150)
plt.close()
print("Saved plots to results/")
