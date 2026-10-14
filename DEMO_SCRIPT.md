# 3-Minute Demo Script — IRIS: Interference Recognition & Intelligence System

---

## SCENE 1: Introduction (15 seconds)

**[SCREEN: Dashboard home page, dark/red theme visible]**

> "I'm presenting **IRIS — Interference Recognition & Intelligence System** — an autonomous AI system that detects, classifies, and mitigates radio frequency interference in communication systems like WiFi, LTE, and satellite links. This is a real problem in dense RF environments: airports, industrial sites, and urban areas where multiple devices compete for spectrum."

---

## SCENE 2: Problem Statement (20 seconds)

**[SCREEN: Slide or dashboard with problem text]**

> "**The problem:** RF interference degrades signal quality, but finding the source requires specialized equipment and expertise. Engineers manually analyze spectrum data—a slow, error-prone process.
>
> **Our solution:** A multi-agent AI system powered by IBM watsonx.ai Granite that autonomously monitors signals, diagnoses interference types, applies fixes, and learns from past incidents."

---

## SCENE 3: Live Demo — Narrowband CW Interference (50 seconds)

**[SCREEN: Dashboard — Simulate panel]**

> "Let me show you a live demo. I'll inject **narrowband CW tone interference** — a common issue from faulty oscillators or GPS jammers."

**[ACTION: Select "Narrowband CW Tone", click "RUN FULL AGENT PIPELINE"]**

**[SCREEN: Watch the Agent Activity panel fill with reasoning trace entries]**

> "Watch the **Agent Activity panel** — this is the autonomous reasoning trace. The system:
> 
> 1. **MonitorAgent** detects an anomaly — the autoencoder reconstruction error is 5× above the threshold.
> 2. **DiagnosisAgent** classifies it: GBM says *narrowband_cw* with 94% confidence, CNN agrees at 91%. High spectral kurtosis (68) and a sharp PSD peak at -0.33 normalised frequency confirm it's a single-tone jammer.
> 3. The agent queries our **knowledge base** via RAG — retrieves docs on CW interference sources and mitigation strategies.
> 4. **MitigationAgent** applies an IIR notch filter at the detected frequency.
> 5. It **closes the loop**: re-runs detection on the filtered signal. SNR improves by **+17.2 dB**. Anomaly score drops from 0.008 to 0.001 — below threshold.
> 6. **ReportAgent** writes a grounded incident report and stores it to memory for future reference."

**[SCREEN: Classification card shows type + confidence, Mitigation panel shows before/after SNR]**

---

## SCENE 4: Technical Stack & Architecture (30 seconds)

**[SCREEN: Architecture diagram or slide]**

> "**How it works:**
>
> - **Data:** 2,000 synthetic IQ signals across 5 interference classes — clean OFDM, narrowband CW, wideband noise, pulsed radar-like, and adjacent-channel leakage.
> - **Models:** Three models work together:
>   - **Autoencoder** (anomaly detector, trained on clean signals only)
>   - **Gradient Boosted Machine** on 12 DSP features: spectral flatness, PAPR, duty cycle, kurtosis, occupied bandwidth — all physically meaningful RF metrics.
>   - **CNN** on spectrograms — learns visual patterns (CW is a vertical line, pulsed is horizontal stripes).
> - **Agents:** Four specialised agents coordinate via an Orchestrator:
>   - **Monitor** → auto-trigger on threshold
>   - **Diagnosis** → classify + RAG + self-verify (re-runs if confidence < 65%)
>   - **Mitigation** → apply DSP fix + closed-loop verification
>   - **Report** → persist to incident memory
> - **LLM:** IBM watsonx.ai **Granite-13b-chat** synthesises explanations grounded in tool outputs and the knowledge base. It never invents measurements.
> - **RAG:** ChromaDB vector store over our RF knowledge base."

---

## SCENE 5: Key Differentiators (20 seconds)

**[SCREEN: Dashboard showing reasoning trace or memory panel]**

> "**What makes this truly agentic:**
>
> 1. **Autonomous**: No user prompt needed. The agent triggers on anomaly score alone, plans its own steps.
> 2. **Self-verification**: If models disagree or confidence is low, it re-measures and asks Granite for a second opinion.
> 3. **Closed-loop**: It doesn't just recommend a fix — it applies it, re-runs detection, and measures the actual before/after SNR improvement.
> 4. **Memory**: Past incidents are stored. The agent refers to them when diagnosing new signals.
> 5. **Grounded answers only**: Every claim traces to a tool output or KB doc. No hallucinations."

---

## SCENE 6: Real-World Applicability (15 seconds)

**[SCREEN: README section on real-world validation]**

> "All data here is synthetic — generated via OFDM/AWGN/jamming models. But the **feature engineering** is physically grounded: Welch PSD, STFT, spectral flatness, PAPR. These transfer directly to real SDR captures from an RTL-SDR, USRP, or HackRF.
>
> **Deployment**: The app is packaged as a Docker container and deployed to **IBM Cloud Code Engine** with cloud-side builds — no Docker Desktop needed locally."

---

## SCENE 7: Closing (10 seconds)

**[SCREEN: Dashboard KPI row or final slide]**

> "In summary: an autonomous, explainable, end-to-end RF interference detection system — from signal to diagnosis to mitigation — powered by IBM watsonx.ai Granite and designed for real-world deployments. Thank you."

---

**TOTAL TIME: 3 minutes**

---

## Visual Flow (Slide/Screen Order)
1. Dashboard home (dark red theme, status: LIVE)
2. Problem statement (text or slide)
3. Simulate panel → click → Agent Activity panel fills
4. Classification card + Mitigation panel (before/after SNR)
5. Architecture diagram
6. Reasoning trace zoom
7. README excerpt on real-world validation
8. Final dashboard overview

---

## Backup: Second Demo (Pulsed Interference) — if time permits

**[ACTION: Select "Pulsed / Radar", click RUN]**

> "Here's pulsed interference — simulates weather radar or a microwave oven. Notice the **high PAPR** (28 dB) and **duty cycle 0.47**. The spectrogram shows horizontal bright stripes during bursts. The agent applies **pulse blanking**: zeros out samples above the 90th percentile. SNR improves by +6.8 dB. It's less effective than the notch filter because more data is discarded, but it's the correct strategy for pulsed interference."
