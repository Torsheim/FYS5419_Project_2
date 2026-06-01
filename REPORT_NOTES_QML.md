# Report notes for Alternative 2: Quantum Machine Learning

These notes are intended as a starting point for the scientific report. Rewrite them in your own style and connect them to your actual numerical results.

## Model definition

For a sample `x = (x_0, ..., x_{p-1})`, the code uses one qubit per feature. The default feature map prepares

```text
|0...0> -> product_j Rz(2π x_j) H |0>_j
```

where all features are first scaled to `[0, 1]`. The default ansatz repeats the block

```text
Ry(theta_0) ... Ry(theta_{p-1})
CNOT chain
Ry(theta_p) ... Ry(theta_{2p-1})
CNOT chain
```

for the requested number of layers. The prediction is

```text
f(x; theta) = P(measured qubit = 1).
```

## Loss function

The implementation uses mean binary cross-entropy,

```text
L(theta) = -1/n sum_i [y_i log f_i + (1-y_i) log(1-f_i)],
```

where `f_i = f(x_i; theta)`. This gives

```text
dL/dtheta_k = 1/n sum_i (f_i-y_i)/(f_i(1-f_i)) df_i/dtheta_k.
```

## Parameter-shift rule

For each parameterized Pauli rotation, the derivative of the model output is computed by

```text
df(x; theta)/dtheta_j = [f(x; theta_j + π/2) - f(x; theta_j - π/2)] / 2.
```

The test suite compares this gradient against a finite-difference gradient.

## Suggested tables and figures

Include a table like this:

| Model | Encoding | Ansatz | Layers | Parameters | Test loss | Test accuracy |
|---|---|---|---:|---:|---:|---:|
| Quantum | h_rz | simple | 2 | ... | ... | ... |
| Quantum | ry_rz | strong | 2 | ... | ... | ... |
| Logistic regression | - | - | - | - | ... | ... |

Include the loss figure generated in `figures/`.

## Discussion points

- The Iris task with only two classes is almost linearly separable, so logistic regression is a strong baseline.
- The quantum model has only a small number of qubits and parameters, so its expressive power depends strongly on the feature map and ansatz.
- Exact state-vector simulation removes shot noise. A real device or shot-based simulator would introduce statistical uncertainty.
- Parameter-shift gradients are exact for these rotation gates, but expensive because each parameter requires two circuit evaluations per gradient step.
