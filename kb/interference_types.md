# RF Interference Types — Knowledge Base

## 1. Clean / Normal Signal (OFDM/WiFi)

**Description:** A healthy Orthogonal Frequency Division Multiplexing (OFDM) signal as used
in WiFi (802.11a/g/n/ac/ax), LTE, and 5G NR.

**Spectral characteristics:**
- Power Spectral Density is relatively flat within the allocated channel bandwidth.
- Sharp roll-off at the channel edges (guard bands).
- Spectral flatness index: ~0.3–0.5 (moderately flat, not pure noise).
- PAPR: typically 9–12 dB due to multi-carrier summation.
- Duty cycle: ~1.0 (signal is always on within the observation window).

**Time-frequency (STFT/spectrogram) appearance:**
- Uniform horizontal band spanning the full channel, consistent across all time slots.

---

## 2. Narrowband / CW Tone Interference

**Description:** A single-frequency continuous wave (CW) sinusoidal signal injected into the
channel. Sources include malfunctioning oscillators, unmodulated carrier leakage, test-equipment
signal generators left on, and deliberate narrowband jammers.

**Spectral characteristics:**
- Extremely sharp spectral spike (approximates a Dirac delta in the PSD).
- Very low spectral flatness (< 0.05) — almost all energy in one bin.
- Very low occupied bandwidth (< 5% of channel).
- High PAPR if CW power exceeds signal power.
- Peak frequency offset identifies the exact jammer frequency.

**Time-frequency appearance:**
- A single bright vertical line at the jammer frequency across all time.

**Real-world sources:**
- Faulty microwave oven magnetron harmonics
- Poorly filtered oscillator in nearby electronics
- Bluetooth advertising packets at fixed frequency
- Rogue CW beacon / drone uplink jammer

---

## 3. Wideband Noise Jamming

**Description:** Flat-spectrum Gaussian noise injected across the entire channel bandwidth or wider.
Effectively raises the noise floor, collapsing the SNR of the desired signal.

**Spectral characteristics:**
- Spectral flatness approaches 1.0 (most noise-like).
- Occupied bandwidth ≥ channel bandwidth (may exceed it).
- Low PAPR (noise has Gaussian amplitude — moderate peaks only).
- Spectral kurtosis ≈ 0 (Gaussian distribution).

**Time-frequency appearance:**
- Uniformly bright spectrogram with no frequency or time structure.

**Real-world sources:**
- Intentional electronic warfare / denial-of-service jammer
- Broadband switching power supply interference
- Poorly shielded computer GPU/CPU radiating broadband EMI
- Industrial arc welders or plasma cutters

---

## 4. Pulsed / Bursty Interference

**Description:** High-power interference that appears in short bursts with a defined duty cycle
(fraction of time the interferer is active). Modelled after radar transmissions, which sweep
high-power pulses across the band.

**Spectral characteristics:**
- High PAPR during the pulse-on period.
- Duty cycle estimate: 0.2–0.8 (bimodal amplitude distribution).
- Spectral kurtosis: elevated (impulsive, heavy-tailed distribution).
- Occupied bandwidth: can be wide (radar chirp) or narrow (fixed-frequency pulse).

**Time-frequency appearance:**
- Horizontal bright stripes appearing only during burst periods; dark gaps between bursts.

**Real-world sources:**
- Airport/military radar (S-band, 2.7–3.5 GHz, overlaps 802.11 2.4 GHz edge)
- Weather radar (C-band, 5.3–5.7 GHz, directly inside WiFi 5 GHz band)
- Pulsed microwave ovens (duty-cycled magnetron)
- Vehicle radar sensors (77 GHz, 24 GHz automotive)

---

## 5. Adjacent-Channel Leakage / Interference (ACI)

**Description:** A legitimate transmitter on an adjacent frequency channel whose sidelobes (due to
imperfect channel filtering or high transmit power) bleed into the channel of interest.

**Spectral characteristics:**
- Asymmetric PSD — one shoulder raised higher than the other.
- Band power ratio ≠ 1.0 (strong imbalance between upper and lower half-bands).
- PSD skewness: non-zero (tail on the side of the neighbouring channel).
- Occupied bandwidth may exceed nominal channel width.

**Time-frequency appearance:**
- Raised noise floor on one edge of the spectrogram, sometimes showing the modulation
  structure of the adjacent transmitter.

**Real-world sources:**
- Overlapping WiFi channels (e.g. channels 6 and 11 near each other)
- Femtocell / small-cell base stations on adjacent LTE resource blocks
- FM/AM broadcast transmitter sidelobe leakage into the cellular band
- DECT cordless phone adjacent to a WiFi AP
