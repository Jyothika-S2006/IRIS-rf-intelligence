# IRIS — Interference Recognition & Intelligence System

> **IRIS — Interference Recognition & Intelligence System** — IBM Hackathon Problem Statement #27
> Stack: IBM watsonx.ai Granite · Python · Streamlit · scikit-learn · PyTorch · Langflow

[![GitHub](https://img.shields.io/badge/GitHub-Jyothika--S2006%2FIRIS--rf--intelligence-red?logo=github)](https://github.com/Jyothika-S2006/IRIS-rf-intelligence)

---

## ⚠️ Data Notice
All RF signal data in this project is **synthetically generated** by `simulator.py`.
Real-world validation would require SDR captures (e.g. RTL-SDR, USRP, HackRF) fed into
`features.py` via the same `extract_features_from_iq()` pipeline. The feature engineering
(Welch PSD, STFT, spectral flatness, PAPR, duty cycle) is physically grounded and would
transfer to real measurements without retraining for gross interference types.

---

## Architecture

```
Signal Simulator → Feature Pipeline → 3 Models (Autoencoder + GBM + CNN)
                                             ↓
                        Multi-Agent System (Orchestrator → Monitor → Diagnosis
                                          → Mitigation → Report)
                                             ↓
                        Langflow Flow (watsonx.ai Granite-13b-chat)
                                             ↓
                        Streamlit Dashboard (premium red/black UI)
                                             ↓
                             http://localhost:8501
```

---

## Project Structure

```
rf_assistant/
├── simulator.py          Signal generator (5 interference classes)
├── features.py           Feature extraction (12 scalar + spectrogram)
├── train_all.py          Full training pipeline (run on Google Colab)
├── agent_tools.py        Tool layer (inference + RAG + memory)
├── agent_system.py       Multi-agent orchestration (4 agents + orchestrator)
├── build_langflow.py     Generates langflow_flow.json
├── dashboard.py          Streamlit UI (premium red/black)
├── data_loader.py        IQ sample loader utility
├── requirements.txt
├── Dockerfile            (ready for future cloud deployment)
├── .streamlit/
│   └── config.toml       Dark theme + red accent
├── kb/
│   ├── interference_types.md
│   ├── mitigation_strategies.md
│   └── interference_sources.md
├── models/
│   ├── gbm_classifier.pkl      ← primary classifier (trained)
│   ├── scaler_gbm.pkl
│   ├── label_encoder.pkl
│   ├── autoencoder.pt          ← anomaly detector (train on Colab)
│   ├── scaler_ae.pkl
│   ├── ae_threshold.json
│   ├── cnn_classifier.pt       ← spectrogram CNN (train on Colab)
│   └── cnn_classes.json
└── data/
    ├── incidents.json           ← persistent agent memory
    ├── metadata.csv
    └── features.csv
```

---

## Quick Start — Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Jyothika-S2006/IRIS-rf-intelligence.git
cd IRIS-rf-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the dashboard
```bash
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser.

### 4. Use the app
- Select an interference type from the **Simulate panel**
- Click **"RUN FULL AGENT PIPELINE"**
- Watch the 4 agents reason live in the **Agent Activity panel**
- See classification, mitigation, before/after SNR in the right panel
- Ask questions in the **Chat box**

---

## Train Models (Google Colab — needed for Autoencoder + CNN)

The GBM classifier is pre-trained and included. To also train the Autoencoder and CNN:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Install
%pip install -q torch torchvision scikit-learn scipy pillow pandas matplotlib seaborn joblib

# Cell 3 — Train
%cd /content/drive/MyDrive/IRIS-rf-intelligence
!python train_all.py
```

Expected runtime: **~10–15 min on Colab T4 GPU**.
Download `models/` folder and paste it back into the repo.

---

## Set watsonx.ai Credentials (optional — enables IBM Granite LLM)

Without credentials the agents run in **deterministic rule-based mode** — fully functional.
To enable Granite natural-language explanations:

```powershell
# Windows PowerShell
$env:WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
$env:WATSONX_APIKEY     = "your-api-key"
$env:WATSONX_PROJECT_ID = "your-project-id"
streamlit run dashboard.py
```

---

## Interference Classes

| Label | Name | Real-World Sources |
|-------|------|--------------------|
| 0 | clean | Normal WiFi / OFDM |
| 1 | narrowband_cw | Faulty oscillator, GPS jammer, microwave harmonic |
| 2 | wideband_noise | Switching PSU EMI, broadband jammer, GPU noise |
| 3 | pulsed | Weather radar, airport radar, microwave oven |
| 4 | adjacent_channel | Overlapping WiFi channels, femtocell, DECT phone |

## Mitigation Strategies

| Interference | Strategy | Expected SNR Gain |
|---|---|---|
| narrowband_cw | IIR notch filter (Q=35) | 10–20 dB |
| wideband_noise | Bandpass filter (60% BW) | 3–10 dB |
| pulsed | Pulse blanking (90th-pct threshold) | 3–7 dB |
| adjacent_channel | Bandpass filter (80% BW) | 5–15 dB |

---

## Multi-Agent System

```
OrchestratorAgent
  ├── MonitorAgent      → analyze_signal()   [autoencoder anomaly detector]
  ├── DiagnosisAgent    → classify_signal() + retrieve_kb() + self-verify
  ├── MitigationAgent   → apply_mitigation() + re_run_detection() [closed loop]
  └── ReportAgent       → store_incident()   [persistent memory]
```

All agent decisions are grounded in:
1. **Tool outputs** — measured anomaly score, classifier probabilities, before/after SNR
2. **RAG KB** — TF-IDF / ChromaDB over `kb/*.md` docs
3. **IBM Granite LLM** — natural-language synthesis (if watsonx credentials set)

---

## Demo

```bash
streamlit run dashboard.py
# Open http://localhost:8501
# Select interference type → RUN FULL AGENT PIPELINE → watch agents reason live
```

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for the full 3-minute presentation script.
See [`JUDGE_QA.md`](JUDGE_QA.md) for 10 likely judge questions with answers.

---

*Built for IBM Hackathon 2024 — Problem Statement #27 | IRIS*
