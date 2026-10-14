"""
simulator.py — RF Signal Simulator
====================================
Generates a labelled dataset of 2 000 IQ samples (complex baseband) across 5 classes:
  0  clean          – WiFi/OFDM-like signal + low AWGN
  1  narrowband     – Continuous-wave (CW) tone jammer
  2  wideband       – Flat-spectrum noise injection (jamming)
  3  pulsed         – Radar-like burst interference (high duty-cycle off-periods)
  4  adjacent       – Adjacent-channel leakage (offset carrier)

Each sample is stored as a .npy file (complex64, length N_SAMPLES) in data/iq/.
A CSV metadata file (data/metadata.csv) records: sample_id, label, label_name,
SNR_dB, interference_power_dBFS, centre_freq_offset_Hz, pulse_duty_cycle.

WHY THIS MATTERS FOR JUDGES
-------------------------------
• OFDM (Orthogonal Frequency Division Multiplexing) is the modulation used by WiFi
  (802.11), LTE and 5G. It spreads data across many closely spaced orthogonal
  subcarriers. Its PSD looks flat and band-limited — easy to tell from jamming.
• CW interference is a single sinusoid: a sharp spike in the PSD.
• Wideband noise fills a wider bandwidth than the desired signal; spectral flatness
  (ratio of geometric to arithmetic mean of PSD bins) approaches 1.
• Pulsed interference shows up as horizontal bands in a spectrogram (STFT) because
  power appears and disappears at a duty cycle.
• Adjacent-channel leakage is the real-world problem of a nearby transmitter whose
  sidelobes bleed into your channel; it looks like an asymmetric PSD shoulder.
"""

import numpy as np
import pandas as pd
import os
import json
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
FS = 20e6          # sample rate: 20 MHz (typical WiFi channel bandwidth)
N  = 1024          # samples per snapshot (short, fast to process)
N_PER_CLASS = 400  # 400 × 5 classes = 2 000 total
SEED = 42
RNG  = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).parent / "data"
IQ_DIR   = DATA_DIR / "iq"
IQ_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "clean",
    1: "narrowband_cw",
    2: "wideband_noise",
    3: "pulsed",
    4: "adjacent_channel",
}

# ── Helper: OFDM-like base signal ────────────────────────────────────────────
def _ofdm_signal(n: int, n_subcarriers: int = 52) -> np.ndarray:
    """
    Simulate one OFDM symbol by IFFT of random QPSK symbols on 52 subcarriers
    (like 802.11a/g in a 20 MHz channel; pilot and guard tones set to 0).
    Returns complex64 time-domain samples, power-normalised to 1.
    """
    freq_domain = np.zeros(n, dtype=complex)
    active = RNG.integers(1, n // 2 - 1, size=n_subcarriers)  # random positive freq bins
    symbols = RNG.choice([-1-1j, -1+1j, 1-1j, 1+1j], size=n_subcarriers)
    freq_domain[active] = symbols
    freq_domain[n - active] = np.conj(symbols)          # Hermitian symmetry → real after IFFT
    x = np.fft.ifft(freq_domain).astype(np.complex64)
    return x / (np.sqrt(np.mean(np.abs(x)**2)) + 1e-12)  # unit power


def _awgn(n: int, snr_db: float) -> np.ndarray:
    """Additive White Gaussian Noise scaled to give the requested SNR (dB)."""
    noise_power = 10 ** (-snr_db / 10)                  # signal power = 1
    return (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex64) * np.sqrt(noise_power / 2)


# ── Class generators ─────────────────────────────────────────────────────────

def gen_clean(n: int = N) -> tuple[np.ndarray, dict]:
    """
    Clean OFDM + mild AWGN (SNR 20–35 dB).
    Baseline class that the anomaly detector trains on.
    """
    snr = float(RNG.uniform(20, 35))
    x   = _ofdm_signal(n) + _awgn(n, snr)
    meta = dict(SNR_dB=snr, interference_power_dBFS=float("nan"),
                centre_freq_offset_Hz=0.0, pulse_duty_cycle=float("nan"))
    return x.astype(np.complex64), meta


def gen_narrowband_cw(n: int = N) -> tuple[np.ndarray, dict]:
    """
    CW tone jammer: a single sinusoid at a random offset frequency injected
    on top of the OFDM signal.  The PSD shows a sharp spike (Delta function).
    Power is randomised ±3 to +15 dB above the signal floor.
    """
    snr         = float(RNG.uniform(10, 25))
    int_power   = float(RNG.uniform(-3, 15))   # dBFS relative to signal
    freq_offset = float(RNG.uniform(-8e6, 8e6))  # Hz within the 20 MHz band

    t   = np.arange(n, dtype=np.float32) / FS
    cw  = np.exp(1j * 2 * np.pi * freq_offset * t).astype(np.complex64)
    cw  = cw * np.sqrt(10 ** (int_power / 10))   # scale to desired power

    x   = _ofdm_signal(n) + _awgn(n, snr) + cw
    meta = dict(SNR_dB=snr, interference_power_dBFS=int_power,
                centre_freq_offset_Hz=freq_offset, pulse_duty_cycle=float("nan"))
    return x.astype(np.complex64), meta


def gen_wideband_noise(n: int = N) -> tuple[np.ndarray, dict]:
    """
    Wideband noise jammer: flat-spectrum Gaussian noise much wider than the
    desired channel.  Spectral flatness approaches 1.0; SNR collapses.
    """
    snr       = float(RNG.uniform(0, 15))
    int_power = float(RNG.uniform(5, 20))

    noise = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex64)
    noise = noise * np.sqrt(10 ** (int_power / 10) / 2)

    x = _ofdm_signal(n) + _awgn(n, snr) + noise
    meta = dict(SNR_dB=snr, interference_power_dBFS=int_power,
                centre_freq_offset_Hz=0.0, pulse_duty_cycle=float("nan"))
    return x.astype(np.complex64), meta


def gen_pulsed(n: int = N) -> tuple[np.ndarray, dict]:
    """
    Pulsed (radar-like) interference: high-power bursts with randomised
    duty cycle (20–80 %) and pulse width.  A spectrogram (STFT) shows
    horizontal bands that appear and disappear over time.
    """
    snr        = float(RNG.uniform(5, 20))
    int_power  = float(RNG.uniform(5, 20))
    duty_cycle = float(RNG.uniform(0.2, 0.8))
    freq_offset = float(RNG.uniform(-5e6, 5e6))

    pulse_len = max(1, int(n * duty_cycle))
    pulse_start = RNG.integers(0, n - pulse_len)

    t   = np.arange(n, dtype=np.float32) / FS
    cw  = np.exp(1j * 2 * np.pi * freq_offset * t).astype(np.complex64)
    envelope = np.zeros(n, dtype=np.float32)
    envelope[pulse_start : pulse_start + pulse_len] = np.sqrt(10 ** (int_power / 10))
    pulse = (cw * envelope).astype(np.complex64)

    x = _ofdm_signal(n) + _awgn(n, snr) + pulse
    meta = dict(SNR_dB=snr, interference_power_dBFS=int_power,
                centre_freq_offset_Hz=freq_offset, pulse_duty_cycle=duty_cycle)
    return x.astype(np.complex64), meta


def gen_adjacent_channel(n: int = N) -> tuple[np.ndarray, dict]:
    """
    Adjacent-channel leakage: an OFDM signal centred at an offset that places
    its spectral shoulder inside our channel.  The PSD asymmetry is distinctive —
    one side has a raised noise floor from the neighbour's sidelobes.
    """
    snr        = float(RNG.uniform(10, 25))
    int_power  = float(RNG.uniform(3, 15))
    # Offset is 1–2 channel widths away but with wide sidelobes overlapping
    freq_offset = float(RNG.choice([-1, 1]) * RNG.uniform(8e6, 18e6))

    neighbour = _ofdm_signal(n) * np.sqrt(10 ** (int_power / 10))
    t         = np.arange(n, dtype=np.float32) / FS
    neighbour = (neighbour * np.exp(1j * 2 * np.pi * freq_offset * t)).astype(np.complex64)

    x = _ofdm_signal(n) + _awgn(n, snr) + neighbour
    meta = dict(SNR_dB=snr, interference_power_dBFS=int_power,
                centre_freq_offset_Hz=freq_offset, pulse_duty_cycle=float("nan"))
    return x.astype(np.complex64), meta


# ── Dataset generation ────────────────────────────────────────────────────────
GENERATORS = {
    0: gen_clean,
    1: gen_narrowband_cw,
    2: gen_wideband_noise,
    3: gen_pulsed,
    4: gen_adjacent_channel,
}

def generate_dataset(n_per_class: int = N_PER_CLASS, verbose: bool = True) -> pd.DataFrame:
    """
    Generate the full labelled dataset.
    Returns a DataFrame with one row per sample (metadata + file path).
    """
    records = []
    total = len(GENERATORS) * n_per_class
    idx = 0

    for label, gen_fn in GENERATORS.items():
        for i in range(n_per_class):
            iq, meta = gen_fn()
            sample_id = f"{CLASS_NAMES[label]}_{i:04d}"
            fpath = IQ_DIR / f"{sample_id}.npy"
            np.save(fpath, iq)

            records.append({
                "sample_id": sample_id,
                "label": label,
                "label_name": CLASS_NAMES[label],
                "file_path": str(fpath),
                **meta,
            })
            idx += 1
            if verbose and idx % 200 == 0:
                print(f"  Generated {idx}/{total} samples …")

    df = pd.DataFrame(records)
    meta_path = DATA_DIR / "metadata.csv"
    df.to_csv(meta_path, index=False)
    if verbose:
        print(f"\n✓ Dataset saved → {meta_path}")
        print(df["label_name"].value_counts().to_string())
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("RF Signal Simulator — generating 2 000 samples")
    print("=" * 60)
    df = generate_dataset(verbose=True)
    # Quick sanity check: load one sample and print shape/dtype
    sample = np.load(df.iloc[0]["file_path"])
    print(f"\nSanity check: shape={sample.shape}, dtype={sample.dtype}")
    print("  Real range: [{:.4f}, {:.4f}]".format(sample.real.min(), sample.real.max()))
    print("  Imag range: [{:.4f}, {:.4f}]".format(sample.imag.min(), sample.imag.max()))
    print("\nDone.")
