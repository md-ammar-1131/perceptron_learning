# Neural Network from Scratch: Perceptron, MLP and Backpropagation

A NumPy-only implementation of Rosenblatt's perceptron and a multilayer perceptron (MLP) with
manually derived backpropagation, built to understand how neural networks learn. No deep learning
framework is used. Written alongside the course *Artificial Neural Networks and Deep Learning
(CS504)* at IIT Ropar.

## What is implemented

| Component | File | Notes |
|---|---|---|
| Rosenblatt's perceptron | `nn/perceptron.py` | Step activation, perceptron learning rule |
| MLP with backpropagation | `nn/mlp.py` | Any number of hidden layers, mini-batch SGD |
| Activations | `nn/activations.py` | sigmoid, tanh, ReLU (and softmax output) |
| Loss | `nn/losses.py` | Cross-entropy |
| Gradient checking | `gradient_check.py` | Compares backprop with numerical gradients |
| XOR experiment | `train_xor.py` | Perceptron fails on XOR, MLP solves it |
| Digit classification | `train_digits.py` | scikit-learn `digits` dataset or MNIST |

## How backpropagation is done here

Forward, for each layer `i`: `z_i = a_{i-1} W_i + b_i`, `a_i = f(z_i)`. The last layer uses softmax and
the loss is mean cross-entropy. All activations are stored during the forward pass.

Backward (`MLP.backward`):

```
delta_L     = (probs - y_onehot) / m          # softmax + cross-entropy gradient
dW_i        = a_{i-1}^T  delta_i
db_i        = sum over the batch of delta_i
delta_{i-1} = (delta_i W_i^T) * f'(a_{i-1})
```

`gradient_check.py` verifies these formulas numerically with a centered difference
`(L(w+eps) - L(w-eps)) / (2 eps)` and reports relative errors around 1e-10 for sigmoid, tanh and ReLU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python gradient_check.py                      # verify backpropagation
python train_xor.py                           # perceptron vs MLP on XOR
python train_digits.py --dataset digits       # quick run, no download
python train_digits.py --dataset mnist --hidden 128 --epochs 20   # MNIST (downloads once)
```

Plots are saved in `results/`.

## Results

**Gradient check** (relative error, lower is better): sigmoid 8e-10, tanh 1e-10, ReLU 1e-10.

**XOR**

| Model | Accuracy |
|---|---|
| Perceptron on AND (linearly separable) | 1.00, converges in 4 epochs |
| Perceptron on XOR | 0.50, never converges |
| MLP `[2, 4, 2]`, tanh | 1.00 |

![MLP solves XOR](results/mlp_xor.png)

**scikit-learn digits (8x8), MLP `[64, 64, 32, 10]`, tanh, 40 epochs, lr 0.1:** test accuracy 0.964
(1259 train / 179 validation / 359 test samples, seed 0).

![Digits training curves](results/digits_curves.png)

## Project structure

```
nn/
  perceptron.py   activations.py   losses.py   mlp.py
gradient_check.py
train_xor.py
train_digits.py
results/          saved plots
```

## Possible extensions

- Add momentum, Nesterov momentum, RMSProp and Adam (planned: optimizer comparison project)
- Add L2 regularization and dropout
- Compare activation functions and network depth on MNIST
