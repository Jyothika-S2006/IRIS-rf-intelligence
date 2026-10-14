"""
features.py — RF Feature Extraction Pipeline
===============================================
Extracts a rich set of features from each IQ snapshot for two purposes:
  1. Tabular feature vector  → GBM classifier + Autoencoder anomaly detector
  2. Spectrogram image (64×64 uint8) → CNN classifier input

FEATURES EXTRACTED (12 scalar features)
-----------------------------------------
  psd_peak_power_db   – Peak power spectral density (dBFS)
  psd_mean_power_db   – Mean PSD across the band
  spectral_flatness   – Wiener entropy: geometric/arithmetic mean of PSD.
                        ≈ 0 for a tone (spiky), ≈ 1 for white noise (flat).
  occupied_bandwidth  – Bandwidth containing 99% of signal power (fraction of FS)
  papr_db             – Peak-to-Average Power Ratio of time-domain signal
                        (high for pulsed/CW, moderate for OFDM).
  duty_cycle_est      – Fraction of samples above 75th percentile power;
                        pulsed interference has a bimodal amplitude distribution.
  spectral_kurtosis   – 4th standardised moment of PSD; impulsive signals create
                        heavy tails (used by professional RF monitors).
  peak_freq_offset    – Normalised frequency of PSD peak (identifies CW tone).
  psd_skewness        – Skewness of PSD bins; asymmetric for adjacent-channel.
  band_power_ratio    – Upper-half / lower-half band power (asymmetry detector).
  rms_amplitude       – RMS of the complex envelope.
  freq_std            – Standard deviation of power-weighted frequency (Hz/FS).

WHY THESE FOR JUDGES
---------------------
• Spectral flatness separates wideband noise (≈1) from OFDM (≈0.4) from CW (≈0).
• PAPR separates pulsed (very high when pulse is on) from clean OFDM (~10 dB).
• Duty cycle directly measures the pulsed fraction.
• Peak frequency offset identifies a CW tone (single dominant bin near non-DC).
• PSD skewness / band power ratio flags adjacent-channel leakage (one-sided shoulder).
• Spectral kurtosis is a standard RF impulsive-noise metric (Agilent VSA, etc.).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, stft as scipy_stft
from scipy.stats import kurtosis, skew
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants (must match simulator.py) ──────────────────────────────────────
FS        = 20e6
N         = 1024
NPERSEG   = 128
NOVERLAP  = 64
SPEC_SIZE = 64       # spectrogram image side length in pixels

DATA_DIR  = Path(__file__).parent / "data"
IQ_DIR    = DATA_DIR / "iq"
SPEC_DIR  = DATA_DIR / "spectrograms"
FEAT_PATH = DATA_DIR / "features.csv"


def _ensure_dirs():
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Core DSP ──────────────────────────────────────────────────────────────────

def compute_welch_psd(iq: np.ndarray):
    """
    Welch's method: average overlapping periodograms to reduce variance.
    Returns (freqs_normalised, psd_linear).  Double-sided (return_onesided=False)
    because complex IQ signals carry information on both positive and negative
    frequencies — unlike a real-valued ADC output.
    """
    f, pxx = welch(iq, fs=1.0, nperseg=NPERSEG, noverlap=NOVERLAP,
                   return_onesided=False, detrend=False)
    f   = np.fft.fftshift(f)
    pxx = np.fft.fftshift(np.abs(pxx))
    return f.astype(np.float64), pxx.astype(np.float64)


def compute_stft_spectrogram(iq: np.ndarray, size: int = SPEC_SIZE) -> np.ndarray:
    """
    STFT → 2-D power spectrogram resized to size×size uint8.

    The STFT reveals WHEN interference occurs (time axis) AND at which frequency.
    This is why pulsed interference shows as horizontal stripes — the burst fills
    the full band but only for a fraction of the time window.
    Returns uint8 array [0..255] suitable for CNN.
    """
    _, _, Zxx = scipy_stft(iq, fs=1.0, nperseg=NPERSEG, noverlap=NOVERLAP,
                           return_onesided=False)
    mag = np.abs(np.fft.fftshift(Zxx, axes=0))
    # Convert to dB with a 60 dB dynamic range floor
    mag_db = 20 * np.log10(mag + 1e-12)
    floor  = mag_db.max() - 60.0
    mag_db = np.clip(mag_db, floor, mag_db.max())
    mag_norm = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-12)

    # Resize to SPEC_SIZE × SPEC_SIZE via PIL (available in base Python/Colab)
    try:
        from PIL import Image
        img = Image.fromarray((mag_norm * 255).astype(np.uint8))
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except ImportError:
        # Fallback: crude row/col subsampling if PIL is missing
        r = np.linspace(0, mag_norm.shape[0] - 1, size, dtype=int)
        c = np.linspace(0, mag_norm.shape[1] - 1, size, dtype=int)
        return (mag_norm[np.ix_(r, c)] * 255).astype(np.uint8)


# ── Scalar feature extraction ─────────────────────────────────────────────────

def extract_features(iq: np.ndarray) -> dict:
    """
    Return dict of 12 scalar features from a complex IQ snapshot.
    All frequency values are normalised to [−0.5, 0.5] (fraction of FS).
    """
    f, pxx = compute_welch_psd(iq)
    pxx   += 1e-20  # numerical floor

    # 1 – Peak PSD (dBFS)
    psd_peak_db = float(10 * np.log10(pxx.max()))

    # 2 – Mean PSD (dBFS)
    psd_mean_db = float(10 * np.log10(pxx.mean()))

    # 3 – Spectral flatness (Wiener entropy) ∈ [0, 1]
    log_mean         = float(np.mean(np.log(pxx)))
    arith_mean       = float(pxx.mean())
    spectral_flatness = float(np.exp(log_mean) / (arith_mean + 1e-20))

    # 4 – Occupied bandwidth (fraction of FS containing 99% power)
    sorted_idx = np.argsort(pxx)[::-1]
    cum_power  = np.cumsum(pxx[sorted_idx]) / pxx.sum()
    n_99       = int(np.searchsorted(cum_power, 0.99)) + 1
    occupied_bw = float(n_99 / len(pxx))

    # 5 – PAPR (dB): peak / mean of instantaneous power
    env     = np.abs(iq) ** 2
    papr_db = float(10 * np.log10(env.max() / (env.mean() + 1e-20)))

    # 6 – Duty cycle estimate: fraction above 75th-percentile power
    thr            = float(np.percentile(env, 75))
    duty_cycle_est = float(np.mean(env > thr))

    # 7 – Spectral kurtosis (Fisher definition: 0 = Gaussian)
    spec_kurtosis = float(kurtosis(pxx, fisher=True))

    # 8 – Peak frequency offset (normalised)
    peak_freq_offset = float(f[np.argmax(pxx)])

    # 9 – PSD skewness
    psd_skewness = float(skew(pxx))

    # 10 – Band power ratio (upper half / lower half)
    mid              = len(pxx) // 2
    band_power_ratio = float(pxx[mid:].sum() / (pxx[:mid].sum() + 1e-20))

    # 11 – RMS amplitude
    rms_amplitude = float(np.sqrt(np.mean(np.abs(iq) ** 2)))

    # 12 – Power-weighted frequency std (normalised)
    weights  = pxx / pxx.sum()
    f_mean   = float(np.dot(f, weights))
    freq_std = float(np.sqrt(np.dot((f - f_mean) ** 2, weights)))

    return dict(
        psd_peak_power_db  = psd_peak_db,
        psd_mean_power_db  = psd_mean_db,
        spectral_flatness  = spectral_flatness,
        occupied_bandwidth = occupied_bw,
        papr_db            = papr_db,
        duty_cycle_est     = duty_cycle_est,
        spectral_kurtosis  = spec_kurtosis,
        peak_freq_offset   = peak_freq_offset,
        psd_skewness       = psd_skewness,
        band_power_ratio   = band_power_ratio,
        rms_amplitude      = rms_amplitude,
        freq_std           = freq_std,
    )


# Canonical column order (used by all model code to keep array columns stable)
FEATURE_COLS = [
    "psd_peak_power_db", "psd_mean_power_db", "spectral_flatness",
    "occupied_bandwidth", "papr_db", "duty_cycle_est", "spectral_kurtosis",
    "peak_freq_offset", "psd_skewness", "band_power_ratio",
    "rms_amplitude", "freq_std",
]


# ── Dataset-level extraction ──────────────────────────────────────────────────

def build_feature_dataset(
    metadata_csv: Path = None,
    save_spectrograms: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Iterate over every sample in the metadata CSV, extract features + spectrogram,
    and save a merged features.csv.  Called from train_all.py on Colab.
    """
    _ensure_dirs()
    if metadata_csv is None:
        metadata_csv = DATA_DIR / "metadata.csv"
    meta  = pd.read_csv(metadata_csv)
    rows  = []
    total = len(meta)

    for i, row in meta.iterrows():
        iq    = np.load(row["file_path"])
        feats = extract_features(iq)
        feats["sample_id"] = row["sample_id"]
        rows.append(feats)

        if save_spectrograms:
            spec = compute_stft_spectrogram(iq)
            np.save(SPEC_DIR / f"{row['sample_id']}.npy", spec)

        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} samples processed …")

    feat_df = pd.DataFrame(rows)
    out_df  = meta.merge(feat_df, on="sample_id")
    out_df.to_csv(FEAT_PATH, index=False)
    if verbose:
        print(f"\n✓ Feature dataset → {FEAT_PATH}  shape={out_df.shape}")
    return out_df


def load_feature_dataset() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        raise FileNotFoundError("Run train_all.py first to build features.csv")
    return pd.read_csv(FEAT_PATH)


def extract_features_from_iq(iq: np.ndarray) -> np.ndarray:
    """
    Convenience: IQ array → 1-D float32 feature vector in canonical FEATURE_COLS order.
    Used by agent_tools.py and dashboard.py for live inference (no training).
    """
    feats = extract_features(iq)
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float32)


if __name__ == "__main__":
    # Quick smoke test on a single synthetic sample
    rng = np.random.default_rng(0)
    iq  = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
    f   = extract_features(iq)
    print("Smoke test — feature dict:")
    for k, v in f.items():
        print(f"  {k:28s} = {v:.6f}")
    spec = compute_stft_spectrogram(iq)
    print(f"\nSpectrogram shape: {spec.shape}, dtype: {spec.dtype}, "
          f"range: [{spec.min()}, {spec.max()}]")
    print("features.py OK")
