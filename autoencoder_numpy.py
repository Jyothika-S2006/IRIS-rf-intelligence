"""
autoencoder_numpy.py — Pure NumPy Autoencoder (no torch required)
===================================================================
Trains and runs a simple MLP autoencoder using only numpy.
Used as a drop-in replacement when torch is not installed.
Saved as models/autoencoder_np.npz + models/ae_threshold.json
"""

import numpy as np
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def _relu(x):
    return np.maximum(0, x)


def _forward(params, x):
    """Encoder → bottleneck → Decoder forward pass."""
    # Encoder
    h1 = _relu(x @ params["W1"] + params["b1"])
    h2 = _relu(h1 @ params["W2"] + params["b2"])
    z  = _relu(h2 @ params["W3"] + params["b3"])
    # Decoder
    h4 = _relu(z  @ params["W4"] + params["b4"])
    h5 = _relu(h4 @ params["W5"] + params["b5"])
    out = h5 @ params["W6"] + params["b6"]
    return out


def train_autoencoder_numpy(X_clean: np.ndarray,
                             epochs: int = 200,
                             lr: float = 1e-3,
                             batch_size: int = 32,
                             verbose: bool = True) -> dict:
    """
    Train autoencoder on clean-only feature matrix X_clean (n_samples, 12).
    Returns params dict.
    Architecture: 12→64→32→8→32→64→12
    """
    rng = np.random.default_rng(42)
    n_in = X_clean.shape[1]  # 12

    def _init(fan_in, fan_out):
        scale = np.sqrt(2.0 / fan_in)
        return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale

    params = {
        "W1": _init(n_in, 64), "b1": np.zeros(64, dtype=np.float32),
        "W2": _init(64, 32),   "b2": np.zeros(32, dtype=np.float32),
        "W3": _init(32, 8),    "b3": np.zeros(8,  dtype=np.float32),
        "W4": _init(8,  32),   "b4": np.zeros(32, dtype=np.float32),
        "W5": _init(32, 64),   "b5": np.zeros(64, dtype=np.float32),
        "W6": _init(64, n_in), "b6": np.zeros(n_in, dtype=np.float32),
    }
    # Adam state
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = X_clean.shape[0]
    losses = []
    t = 0

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        ep_loss = 0.0

        for start in range(0, n, batch_size):
            batch = X_clean[idx[start:start + batch_size]]
            B = batch.shape[0]
            t += 1

            # ── Forward ──────────────────────────────────────────────────
            h1  = _relu(batch @ params["W1"] + params["b1"])
            h2  = _relu(h1    @ params["W2"] + params["b2"])
            z   = _relu(h2    @ params["W3"] + params["b3"])
            h4  = _relu(z     @ params["W4"] + params["b4"])
            h5  = _relu(h4    @ params["W5"] + params["b5"])
            out = h5 @ params["W6"] + params["b6"]

            # MSE loss
            diff  = out - batch
            loss  = float(np.mean(diff ** 2))
            ep_loss += loss * B

            # ── Backward (chain rule through each layer) ──────────────────
            dout = 2 * diff / (B * n_in)

            dW6 = h5.T @ dout
            db6 = dout.sum(0)
            dh5 = dout @ params["W6"].T
            dh5 *= (h5 > 0)

            dW5 = h4.T @ dh5
            db5 = dh5.sum(0)
            dh4 = dh5 @ params["W5"].T
            dh4 *= (h4 > 0)

            dW4 = z.T @ dh4
            db4 = dh4.sum(0)
            dz  = dh4 @ params["W4"].T
            dz  *= (z > 0)

            dW3 = h2.T @ dz
            db3 = dz.sum(0)
            dh2 = dz @ params["W3"].T
            dh2 *= (h2 > 0)

            dW2 = h1.T @ dh2
            db2 = dh2.sum(0)
            dh1 = dh2 @ params["W2"].T
            dh1 *= (h1 > 0)

            dW1 = batch.T @ dh1
            db1 = dh1.sum(0)

            grads = dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3,
                         W4=dW4,b4=db4,W5=dW5,b5=db5,W6=dW6,b6=db6)

            # ── Adam update ───────────────────────────────────────────────
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** t)
                v_hat = v[k] / (1 - beta2 ** t)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        ep_loss /= n
        losses.append(ep_loss)
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

    return params


def reconstruction_errors(params: dict, X: np.ndarray) -> np.ndarray:
    """Compute per-sample MSE reconstruction error."""
    out = _forward(params, X)
    return np.mean((out - X) ** 2, axis=1).astype(np.float32)


def save_autoencoder(params: dict, threshold: float, n_in: int = 12):
    MODEL_DIR.mkdir(exist_ok=True)
    np.savez_compressed(MODEL_DIR / "autoencoder_np.npz",
                        **{k: v for k, v in params.items()})
    with open(MODEL_DIR / "ae_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "n_in": n_in, "backend": "numpy"}, f)
    print(f"  Saved autoencoder_np.npz  threshold={threshold:.6f}")


def load_autoencoder():
    npz = np.load(MODEL_DIR / "autoencoder_np.npz")
    params = {k: npz[k] for k in npz.files}
    with open(MODEL_DIR / "ae_threshold.json") as f:
        cfg = json.load(f)
    return params, cfg["threshold"]
