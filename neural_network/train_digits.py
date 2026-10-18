"""Train the from-scratch MLP on a handwritten-digit dataset.

Datasets
--------
digits : 8x8 handwritten digits bundled with scikit-learn (1797 samples). Quick, no download.
mnist  : 28x28 MNIST (70000 samples). Downloaded once via scikit-learn (needs internet).

Examples:
    python train_digits.py --dataset digits
    python train_digits.py --dataset mnist --hidden 128 --epochs 20 --lr 0.1
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import MLP


def load_data(name):
    if name == "digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        X, y = data.data / 16.0, data.target
    elif name == "mnist":
        from sklearn.datasets import fetch_openml
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = data.data / 255.0, data.target.astype(int)
    else:
        raise ValueError("dataset must be 'digits' or 'mnist'")
    return X.astype(float), y.astype(int)


def split(X, y, seed=0, val_frac=0.1, test_frac=0.2):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    n_test, n_val = int(len(X) * test_frac), int(len(X) * val_frac)
    return (X[n_test + n_val:], y[n_test + n_val:],
            X[n_test:n_test + n_val], y[n_test:n_test + n_val],
            X[:n_test], y[:n_test])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--activation", default="tanh", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = load_data(args.dataset)
    X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, args.seed)
    print(f"{args.dataset}: train {len(X_tr)}, val {len(X_va)}, test {len(X_te)}")

    sizes = [X.shape[1], *args.hidden, 10]
    model = MLP(sizes, activation=args.activation, seed=args.seed)
    print("Architecture:", sizes, "| activation:", args.activation)
    hist = model.fit(X_tr, y_tr, X_va, y_va, epochs=args.epochs, lr=args.lr,
                     batch_size=args.batch_size, seed=args.seed)

    test_acc = model.accuracy(X_te, y_te)
    print(f"\nTest accuracy: {test_acc:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="validation")
    ax[1].set_title(f"Accuracy (test = {test_acc:.3f})")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    out = f"results/{args.dataset}_curves.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
