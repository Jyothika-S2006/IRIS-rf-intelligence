"""
train_all.py — Full Training Pipeline (run on Google Colab or any GPU/fast CPU)
=================================================================================
Trains all three models and saves them to /models:
  1. Autoencoder  – anomaly detector (trained on CLEAN signals only)
  2. GBM          – gradient-boosted classifier on 12 scalar features (primary)
  3. CNN          – convolutional classifier on 64×64 spectrogram images

Usage
------
  # Full training (2 000 samples, ~10-15 min on Colab T4):
  python train_all.py

  # Smoke test on i3 laptop to verify code before uploading to Colab:
  python train_all.py --smoke

  # Specific modules only:
  python train_all.py --modules autoencoder gbm

Google Colab Instructions (see README for full detail)
-------------------------------------------------------
  1. Upload this entire rf_assistant/ folder to your Google Drive.
  2. Open a new Colab notebook and run:
       from google.colab import drive
       drive.mount('/content/drive')
       %cd /content/drive/MyDrive/rf_assistant
       !pip install -q scikit-learn torch torchvision scipy pillow pandas matplotlib seaborn joblib
  3. Run:
       !python train_all.py
  4. When finished, the models/ folder contains:
       autoencoder.pt, scaler_ae.pkl,
       gbm_classifier.pkl, scaler_gbm.pkl, label_encoder.pkl,
       cnn_classifier.pt, cnn_classes.json
  5. Download models/ and place it inside rf_assistant/ on your laptop.
"""

import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
IQ_DIR     = DATA_DIR / "iq"
SPEC_DIR   = DATA_DIR / "spectrograms"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports"
for d in [MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Shared constants ───────────────────────────────────────────────────────────
CLASS_NAMES  = ["clean", "narrowband_cw", "wideband_noise", "pulsed", "adjacent_channel"]
N_CLASSES    = len(CLASS_NAMES)
FEATURE_COLS = [
    "psd_peak_power_db", "psd_mean_power_db", "spectral_flatness",
    "occupied_bandwidth", "papr_db", "duty_cycle_est", "spectral_kurtosis",
    "peak_freq_offset", "psd_skewness", "band_power_ratio",
    "rms_amplitude", "freq_std",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 0. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(smoke: bool = False):
    """
    Generate IQ samples (if not already on disk), then extract features
    and spectrograms. Returns the feature DataFrame.
    In smoke mode: 50 samples/class → 250 total (runs in ~30 s on any machine).
    """
    n_per_class = 25 if smoke else 400
    print(f"\n{'='*60}")
    print(f"STEP 0 — Data Preparation  (smoke={smoke}, {n_per_class}/class)")
    print(f"{'='*60}")

    meta_path = DATA_DIR / "metadata.csv"
    if smoke or not meta_path.exists():
        # Always regenerate in smoke mode so sizes are consistent
        from simulator import generate_dataset
        df_meta = generate_dataset(n_per_class=n_per_class, verbose=True)
    else:
        df_meta = pd.read_csv(meta_path)
        print(f"✓ Existing dataset loaded: {len(df_meta)} samples")

    feat_path = DATA_DIR / "features.csv"
    if smoke or not feat_path.exists():
        from features import build_feature_dataset
        df_feat = build_feature_dataset(
            metadata_csv=meta_path,
            save_spectrograms=True,
            verbose=True,
        )
    else:
        df_feat = pd.read_csv(feat_path)
        print(f"✓ Existing features loaded: {df_feat.shape}")

    return df_feat


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AUTOENCODER ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(df: pd.DataFrame, smoke: bool = False):
    """
    Trains a fully-connected autoencoder on CLEAN signals only.
    At inference time, a signal with high reconstruction error is flagged
    as anomalous (interference present).

    Architecture:
      Encoder: 12 → 64 → 32 → 8
      Decoder: 8 → 32 → 64 → 12
    Loss: MSE on the normalised feature vector.
    Threshold: 95th percentile reconstruction error on the CLEAN training set
               (i.e., only 5% of clean signals will be false-positives).

    Why an autoencoder?
    The autoencoder learns the *manifold* of normal (clean) signals. It cannot
    reconstruct interference patterns well because it was never trained on them,
    so their reconstruction error is high — a classic one-class / novelty detector.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*60}")
    print("STEP 1 — Autoencoder Anomaly Detector")
    print(f"{'='*60}")
    t0 = time.time()

    # Features for clean class only
    clean_df = df[df["label"] == 0][FEATURE_COLS].dropna()
    X_clean  = clean_df.values.astype(np.float32)

    scaler = StandardScaler()
    X_clean_s = scaler.fit_transform(X_clean)

    # Also prepare ALL data for threshold/eval
    X_all = scaler.transform(
        df[FEATURE_COLS].fillna(0).values.astype(np.float32))
    y_all = df["label"].values

    # DataLoader
    tensor_clean = torch.FloatTensor(X_clean_s)
    loader = DataLoader(
        TensorDataset(tensor_clean, tensor_clean),
        batch_size=32, shuffle=True)

    # Model
    class Autoencoder(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_in, 64), nn.ReLU(),
                nn.Linear(64, 32),   nn.ReLU(),
                nn.Linear(32, 8),    nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 32),    nn.ReLU(),
                nn.Linear(32, 64),   nn.ReLU(),
                nn.Linear(64, n_in),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    n_in   = X_clean_s.shape[1]
    model  = Autoencoder(n_in)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 20 if smoke else 100

    print(f"  Training autoencoder: {epochs} epochs on {len(X_clean_s)} clean samples ...")
    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, _ in loader:
            out  = model(xb)
            loss = loss_fn(out, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_clean_s)
        losses.append(ep_loss)
        if ep % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep:3d}/{epochs}  loss={ep_loss:.6f}")

    # Reconstruction error on clean training set → set threshold
    model.eval()
    with torch.no_grad():
        X_t      = torch.FloatTensor(X_clean_s)
        recon    = model(X_t).numpy()
        errors_clean = np.mean((recon - X_clean_s) ** 2, axis=1)
    threshold = float(np.percentile(errors_clean, 95))
    print(f"  Anomaly threshold (95th pct of clean): {threshold:.6f}")

    # Evaluate on all classes
    model.eval()
    with torch.no_grad():
        X_all_t  = torch.FloatTensor(X_all)
        recon_all = model(X_all_t).numpy()
        errors_all = np.mean((recon_all - X_all) ** 2, axis=1)
    y_pred_anomaly = (errors_all > threshold).astype(int)
    # Anomaly = any label != 0 (clean)
    y_true_anomaly = (y_all != 0).astype(int)

    from sklearn.metrics import classification_report
    print("\n  Anomaly Detection Report (0=clean, 1=interference):")
    print(classification_report(y_true_anomaly, y_pred_anomaly,
                                target_names=["clean", "interference"], digits=3))

    if HAS_PLOT:
        plt.figure(figsize=(7, 3))
        plt.plot(losses, color="#E50914")
        plt.title("Autoencoder Training Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "autoencoder_loss.png", dpi=100)
        plt.close()

        plt.figure(figsize=(9, 4))
        for lbl, name in enumerate(CLASS_NAMES):
            idx = y_all == lbl
            plt.hist(errors_all[idx], bins=30, alpha=0.6, label=name)
        plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
        plt.xlabel("Reconstruction Error"); plt.ylabel("Count")
        plt.title("Autoencoder Reconstruction Error by Class")
        plt.legend(); plt.tight_layout()
        plt.savefig(REPORT_DIR / "autoencoder_errors.png", dpi=100)
        plt.close()

    # Save model + scaler + threshold
    torch.save(model.state_dict(), MODEL_DIR / "autoencoder.pt")
    joblib.dump(scaler, MODEL_DIR / "scaler_ae.pkl")
    with open(MODEL_DIR / "ae_threshold.json", "w") as fh:
        json.dump({"threshold": threshold, "n_in": n_in}, fh)

    print(f"\n✓ Autoencoder saved → models/autoencoder.pt  ({time.time()-t0:.1f}s)")
    return model, scaler, threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GBM CLASSIFIER (Primary)
# ═══════════════════════════════════════════════════════════════════════════════

def train_gbm(df: pd.DataFrame, smoke: bool = False):
    """
    Gradient Boosted Machine on 12 engineered features.
    Uses HistGradientBoostingClassifier (scikit-learn ≥ 0.24):
      - Histogram-based splits (fast, handles NaN natively).
      - Native multi-class support.
      - No GPU needed; trains in seconds even on CPU.

    Why GBM over a plain decision tree or SVM?
    GBMs handle non-linear feature interactions (e.g., high PAPR AND duty_cycle
    together = pulsed), are robust to feature scale, and give class probabilities
    via predict_proba — the confidence score shown on the dashboard.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)

    print(f"\n{'='*60}")
    print("STEP 2 — GBM Classifier (primary)")
    print(f"{'='*60}")
    t0 = time.time()

    df_clean = df[FEATURE_COLS + ["label", "label_name"]].dropna()
    X = df_clean[FEATURE_COLS].values.astype(np.float32)
    # Fit on string label_name so le.classes_ = ['adjacent_channel', 'clean', ...]
    y_str = df_clean["label_name"].values
    y_int = df_clean["label"].values

    le = LabelEncoder()
    le.fit(y_str)
    y_enc = le.transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    max_iter = 50 if smoke else 300
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1,
        max_depth=6, random_state=42, verbose=0)

    print(f"  Training GBM: max_iter={max_iter}, train={len(X_train)}, test={len(X_test)} …")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(cm, le.classes_, "GBM Classifier", "gbm_confusion.png")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\n  Feature Importances:")
        print(imp_df.to_string(index=False))
        imp_df.to_csv(REPORT_DIR / "gbm_feature_importance.csv", index=False)

    # Save
    joblib.dump(model,  MODEL_DIR / "gbm_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_gbm.pkl")
    joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")

    print(f"\n✓ GBM saved → models/gbm_classifier.pkl  ({time.time()-t0:.1f}s)")
    return model, scaler, le


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CNN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def train_cnn(df: pd.DataFrame, smoke: bool = False):
    """
    Small CNN on 64×64 spectrogram images.

    Architecture (deliberately tiny for CPU/Colab T4):
      Conv2d(1→16, 3×3) → BN → ReLU → MaxPool(2×2)
      Conv2d(16→32, 3×3) → BN → ReLU → MaxPool(2×2)
      Conv2d(32→64, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
      FC(64×4×4 → 128) → ReLU → Dropout(0.3) → FC(128 → 5)

    Why CNN on spectrograms?
    Each interference type has a distinctive 2-D visual fingerprint:
      • Clean OFDM: uniform horizontal band (all subcarriers, always on)
      • CW tone: a thin vertical line (one frequency, all time)
      • Wideband noise: uniformly bright full-band spectrogram
      • Pulsed: bright stripes only during bursts (horizontal banding)
      • Adjacent channel: bright band shifted to one side
    A CNN learns these patterns directly from pixel values — it finds the
    spatial features (edges, blobs) that matter, without manual design.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)
    from sklearn.preprocessing import LabelEncoder

    print(f"\n{'='*60}")
    print("STEP 3 — CNN Classifier (spectrogram images)")
    print(f"{'='*60}")
    t0 = time.time()

    # ── Dataset class ──────────────────────────────────────────────────────
    class SpectrogramDataset(Dataset):
        def __init__(self, sample_ids, labels):
            self.ids    = sample_ids
            self.labels = labels
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, idx):
            path = SPEC_DIR / f"{self.ids[idx]}.npy"
            img  = np.load(path).astype(np.float32) / 255.0  # [0,1]
            img  = torch.FloatTensor(img).unsqueeze(0)        # [1, H, W]
            return img, int(self.labels[idx])

    # ── Data split ─────────────────────────────────────────────────────────
    # Check all spectrogram files exist
    valid_mask = df["sample_id"].apply(
        lambda sid: (SPEC_DIR / f"{sid}.npy").exists())
    df_valid = df[valid_mask].copy()
    if len(df_valid) < len(df):
        print(f"  WARNING: {len(df)-len(df_valid)} spectrograms missing, skipped.")

    le = LabelEncoder()
    le.fit(df_valid["label_name"].values)
    y  = le.transform(df_valid["label_name"].values)
    ids = df_valid["sample_id"].values

    ids_train, ids_test, y_train, y_test = train_test_split(
        ids, y, test_size=0.2, stratify=y, random_state=42)

    bs       = 16 if smoke else 32
    epochs   = 5  if smoke else 30
    n_workers = 0  # 0 = main process (safe on Windows/Colab)

    train_ds = SpectrogramDataset(ids_train, y_train)
    test_ds  = SpectrogramDataset(ids_test,  y_test)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=n_workers)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=n_workers)

    # ── Model ──────────────────────────────────────────────────────────────
    class SmallCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                          # 32×32
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                          # 16×16
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),                                  # 4×4
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model  = SmallCNN(N_CLASSES).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Training CNN: {epochs} epochs, batch={bs}, "
          f"train={len(train_ds)}, test={len(test_ds)} …")

    train_accs, test_accs = [], []
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
        train_acc = correct / total
        sched.step()

        # Validation accuracy
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / vt
        train_accs.append(train_acc); test_accs.append(val_acc)
        if ep % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep:3d}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            all_preds.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    print(f"\n  Test Accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=le.classes_, digits=3))
    cm = confusion_matrix(all_true, all_preds)
    _save_confusion_matrix(cm, le.classes_, "CNN Classifier", "cnn_confusion.png")

    # Training curve
    if HAS_PLOT:
        plt.figure(figsize=(7, 3))
        plt.plot(train_accs, label="Train", color="#E50914")
        plt.plot(test_accs,  label="Val",   color="#8B0000", linestyle="--")
        plt.title("CNN Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(REPORT_DIR / "cnn_accuracy.png", dpi=100)
        plt.close()

    # Save
    torch.save(model.state_dict(), MODEL_DIR / "cnn_classifier.pt")
    with open(MODEL_DIR / "cnn_classes.json", "w") as fh:
        json.dump({"classes": list(le.classes_), "n_classes": N_CLASSES}, fh)

    print(f"\n✓ CNN saved → models/cnn_classifier.pt  ({time.time()-t0:.1f}s)")
    return model, le


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_confusion_matrix(cm, labels, title, fname):
    if not HAS_PLOT:
        print(f"  (Skipping confusion matrix plot — matplotlib not installed)")
        return
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(REPORT_DIR / fname, dpi=100)
    plt.close()
    print(f"  Confusion matrix saved → reports/{fname}")


def _print_banner(text):
    print(f"\n{'#'*60}")
    print(f"  {text}")
    print(f"{'#'*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RF Interference Detection — Training Pipeline")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test mode: tiny dataset, few epochs (run on i3 laptop to verify code)")
    parser.add_argument("--modules", nargs="+",
                        choices=["data", "autoencoder", "gbm", "cnn"],
                        default=["data", "autoencoder", "gbm", "cnn"],
                        help="Which modules to run (default: all)")
    args = parser.parse_args()

    if args.smoke:
        _print_banner("SMOKE TEST MODE — tiny dataset, fast training")
        print("  Purpose: verify all code runs end-to-end before Colab upload.")
        print("  Expected runtime: < 3 minutes on a laptop i3.")
    else:
        _print_banner("FULL TRAINING — 2 000 samples, production models")

    t_total = time.time()

    # ── Step 0: Data ──────────────────────────────────────────────────────
    if "data" in args.modules:
        df = prepare_data(smoke=args.smoke)
    else:
        from features import load_feature_dataset
        df = load_feature_dataset()
        print(f"✓ Loaded existing feature dataset: {df.shape}")

    # ── Step 1: Autoencoder ───────────────────────────────────────────────
    if "autoencoder" in args.modules:
        train_autoencoder(df, smoke=args.smoke)

    # ── Step 2: GBM ──────────────────────────────────────────────────────
    if "gbm" in args.modules:
        train_gbm(df, smoke=args.smoke)

    # ── Step 3: CNN ───────────────────────────────────────────────────────
    if "cnn" in args.modules:
        train_cnn(df, smoke=args.smoke)

    _print_banner(f"ALL DONE — total time: {time.time()-t_total:.1f}s")
    print(f"\n  Saved models:")
    for f_model in sorted(MODEL_DIR.iterdir()):
        size_kb = f_model.stat().st_size / 1024
        print(f"    {f_model.name:35s} {size_kb:8.1f} KB")
    print(f"\n  Reports:")
    for f_rep in sorted(REPORT_DIR.iterdir()):
        print(f"    {f_rep.name}")
    print()

    if args.smoke:
        print("=" * 60)
        print("Smoke test PASSED ✓")
        print("Upload rf_assistant/ to Google Drive and run:")
        print("  python train_all.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
