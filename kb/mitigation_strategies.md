# RF Interference Mitigation Strategies — Knowledge Base

## Overview
Mitigation strategies depend on the interference type. Always verify the fix worked by
re-running detection after applying the mitigation (before/after SNR and anomaly score).

---

## Strategy 1: Notch Filter (for Narrowband / CW Interference)

**Applies to:** Narrowband CW tone interference.

**Mechanism:**
A notch (band-stop) filter is a sharp filter that attenuates a very narrow frequency range
centred on the interfering tone while passing all other frequencies. It is the surgical
removal of a single frequency spike.

**Implementation:**
- Design a 2nd-order IIR notch filter (e.g., scipy.signal.iirnotch) at the detected
  peak frequency offset.
- Filter bandwidth (quality factor Q) should be narrow: Q = 30–50 for a single CW tone.
- Apply as a digital filter to the IQ samples.

**Expected outcome:**
- Anomaly score drops below threshold.
- SNR improves by 10–20 dB when the jammer power was significant.
- Minimal signal degradation (< 0.1 dB loss in adjacent bins).

**When it fails:**
- If the jammer is frequency-hopping (changes frequency rapidly), a static notch will
  not track it. Use adaptive LMS filtering instead.

---

## Strategy 2: Channel Change (for Adjacent-Channel Interference)

**Applies to:** Adjacent-channel leakage / interference (ACI).

**Mechanism:**
Move the desired receiver to a different non-overlapping channel that has no adjacent
interferer. For 2.4 GHz WiFi: channels 1, 6, 11 are the three non-overlapping choices.
For 5 GHz WiFi: many more non-overlapping channels available (36, 40, 44, 48, …).

**Implementation:**
- Scan all available channels and measure the noise floor on each.
- Select the channel with the lowest interference power and highest SNR.
- In simulation: shift the signal centre frequency by the required channel offset.

**Expected outcome:**
- PSD asymmetry disappears.
- Band power ratio returns to ~1.0.
- SNR improvement: 5–15 dB depending on channel separation.

**When it fails:**
- If all channels are congested, channel change is insufficient. Use frequency planning
  with spatial separation or upgrade to 5 GHz / 6 GHz (WiFi 6E).

---

## Strategy 3: Bandpass Filtering (for Wideband Noise Jamming)

**Applies to:** Wideband noise, broadband EMI.

**Mechanism:**
A bandpass filter passes only the desired signal band and rejects out-of-band noise.
It cannot remove in-band noise but limits the noise power integrated over the receiver
bandwidth by rejecting energy outside the signal band.

**Implementation:**
- Design a FIR or IIR bandpass filter with passband = nominal signal bandwidth.
- Stopband attenuation: 40–60 dB.
- Apply to IQ samples before further processing.

**Expected outcome:**
- Reduces total noise power by a factor equal to the ratio of receiver BW to jammer BW.
- SNR improvement: 3–10 dB for a jammer 3–10× wider than the signal.
- Spectral flatness decreases (bandpass-filtered noise is no longer flat across the receiver).

**When it fails:**
- If the jammer is the same bandwidth as the desired signal, bandpass filtering gives no gain.
  Use spread-spectrum techniques (FHSS, DSSS) or MIMO null-steering.

---

## Strategy 4: Pulse Blanking / Gating (for Pulsed Interference)

**Applies to:** Pulsed / radar-like interference.

**Mechanism:**
Detect the pulse envelope (using the duty cycle and PAPR features), then gate (zero out /
blank) the received samples during the pulse-on period. OFDM receivers tolerate some
blanked samples better than continuous interference.

**Implementation:**
- Estimate pulse envelope: compare instantaneous power to a threshold (e.g., 3 dB above
  the estimated clean signal floor).
- Set samples above threshold to zero (hard blanking) or taper them (soft blanking with a
  raised-cosine window to reduce spectral splatter).
- Remaining duty-cycle fraction of samples are interference-free.

**Expected outcome:**
- Effective SNR improves proportionally to (1 − duty_cycle).
- Anomaly score drops as the high-PAPR bursts are removed.
- At duty cycle 0.5: SNR improves by ~3 dB; at 0.2: ~7 dB improvement.

**When it fails:**
- If duty cycle > 0.8, most samples are blanked and the useful data rate collapses.
  Combine blanking with frequency diversity (spread-spectrum).

---

## Strategy 5: Antenna Placement / Shielding (Physical Layer)

**Applies to:** All interference types, especially when the jammer is physically close.

**Mechanism:**
Physical separation, directional antennas, or RF shielding reduce the received interference
power at the source, before any digital processing is needed.

**Implementation recommendations:**
- Move the AP / receiver antenna away from known interferer locations (microwaves, motors).
- Use directional (patch, Yagi) antennas pointed away from the interferer.
- Install EMI shielding enclosures around sensitive electronics.
- Use shielded/twisted-pair cabling for connections between RF components.

**Expected outcome:**
- 10–40 dB reduction in received interference power for every factor of 3 in distance
  (free-space path loss: 20 dB per decade of distance).
- Most cost-effective for permanent installations.

---

## General Mitigation Decision Tree

```
Interference detected?
  ├── Narrowband CW spike?          → Notch Filter
  ├── Wideband flat noise?          → Bandpass Filter + check for broadband EMI source
  ├── Pulsed / bursty?             → Pulse Blanking / Gating
  ├── Adjacent-channel asymmetry? → Channel Change / Frequency Planning
  └── All types (severe):          → Antenna repositioning + physical shielding
```
