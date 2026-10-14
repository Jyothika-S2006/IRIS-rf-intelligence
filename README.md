# IRIS — Interference Recognition & Intelligence System

> **IRIS — Interference Recognition & Intelligence System** — IBM Hackathon Problem Statement #27  
> Stack: IBM watsonx.ai Granite · Langflow · Python · Streamlit · PyTorch · scikit-learn

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
                        IBM Cloud Code Engine (Docker, cloud-side build)
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
├── Dockerfile
├── .dockerignore
├── .streamlit/
│   └── config.toml       Dark theme + red accent
├── kb/
│   ├── interference_types.md
│   ├── mitigation_strategies.md
│   └── interference_sources.md
├── models/               ← created by train_all.py
│   ├── autoencoder.pt
│   ├── scaler_ae.pkl
│   ├── ae_threshold.json
│   ├── gbm_classifier.pkl
│   ├── scaler_gbm.pkl
│   ├── label_encoder.pkl
│   ├── cnn_classifier.pt
│   └── cnn_classes.json
└── data/                 ← created by train_all.py
    ├── metadata.csv
    ├── features.csv
    ├── incidents.json
    ├── iq/
    └── spectrograms/
```

---

## Step-by-Step Setup

### 1. Smoke Test (verify code on your laptop, ~3 min)

```bash
cd rf_assistant
pip install numpy scipy pandas scikit-learn torch pillow joblib
python train_all.py --smoke
```

Expected output: `Smoke test PASSED ✓`

### 2. Full Training on Google Colab

1. **Upload** the entire `rf_assistant/` folder to Google Drive.

2. **Open a new Colab notebook** and run these cells:

```python
# Cell 1 — mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2 — install dependencies
%pip install -q scikit-learn torch torchvision scipy pillow pandas \
               matplotlib seaborn joblib chromadb sentence-transformers \
               ibm-watsonx-ai streamlit plotly
```

```python
# Cell 3 — run full training
%cd /content/drive/MyDrive/rf_assistant
!python train_all.py
```

Expected runtime: **~10–15 min on Colab T4 GPU**.

3. **Download the `models/` folder** from Google Drive to your laptop and place it inside `rf_assistant/`.

### 3. Run the Dashboard Locally

```bash
cd rf_assistant
pip install -r requirements.txt
streamlit run dashboard.py
```

Open http://localhost:8501

### 4. Set watsonx.ai Credentials (optional, enables Granite LLM)

```bash
# Windows PowerShell
$env:WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
$env:WATSONX_APIKEY     = "your-api-key"
$env:WATSONX_PROJECT_ID = "your-project-id"
streamlit run dashboard.py
```

Without credentials the agents run in deterministic (rule-based) mode — fully functional.

---

## IBM Cloud Code Engine Deployment (cloud-side build, no Docker locally)

### Prerequisites
```bash
# Install IBM Cloud CLI
# Windows: winget install IBM.IBMCloudCLI
ibmcloud login --apikey YOUR_APIKEY -r us-south
ibmcloud plugin install code-engine
ibmcloud ce project create --name rf-interference
ibmcloud ce project select --name rf-interference
```

### Option A: Build from a GitHub repository (recommended)
```bash
# 1. Push rf_assistant/ to a public (or private with secret) GitHub repo
# 2. Create an IBM Container Registry namespace
ibmcloud cr namespace-add rf-assistant-ns

# 3. Create a build (cloud-side, no Docker needed locally)
ibmcloud ce build create \
  --name rf-build \
  --image us.icr.io/rf-assistant-ns/rf-assistant:latest \
  --source https://github.com/YOUR_USERNAME/rf-assistant \
  --context-dir rf_assistant \
  --strategy dockerfile \
  --size medium

# 4. Run the build
ibmcloud ce buildrun submit --build rf-build

# 5. Deploy the application
ibmcloud ce application create \
  --name rf-assistant \
  --image us.icr.io/rf-assistant-ns/rf-assistant:latest \
  --port 8080 \
  --min-scale 1 \
  --max-scale 3 \
  --memory 4G \
  --cpu 2 \
  --env WATSONX_URL=https://us-south.ml.cloud.ibm.com \
  --env WATSONX_APIKEY=YOUR_APIKEY \
  --env WATSONX_PROJECT_ID=YOUR_PROJECT_ID

# 6. Get the public URL
ibmcloud ce application get --name rf-assistant | grep URL
```

### Option B: Build from local source (requires IBM Cloud CLI buildpacks, no Docker daemon)
```bash
ibmcloud ce application create \
  --name rf-assistant \
  --build-source . \
  --build-dockerfile Dockerfile \
  --image us.icr.io/rf-assistant-ns/rf-assistant:latest \
  --port 8080 \
  --env WATSONX_APIKEY=YOUR_APIKEY \
  --env WATSONX_PROJECT_ID=YOUR_PROJECT_ID
```

---

## Interference Classes

| Label | Name | Real-World Sources |
|-------|------|--------------------|
| 0 | clean | Normal WiFi/OFDM |
| 1 | narrowband_cw | Faulty oscillator, GPS jammer, microwave harmonic |
| 2 | wideband_noise | Switching PSU EMI, broadband jammer, GPU |
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
  ├── MonitorAgent      → analyze_signal() [autoencoder anomaly detector]
  ├── DiagnosisAgent    → classify_signal() + retrieve_kb() + self-verify
  ├── MitigationAgent   → apply_mitigation() + re_run_detection() [closed loop]
  └── ReportAgent       → store_incident() [persistent memory]
```

All agent decisions are grounded in:
1. **Tool outputs** (measured anomaly score, classifier probabilities, before/after SNR)
2. **RAG KB** (ChromaDB / TF-IDF over `kb/*.md`)
3. **Granite LLM** for natural-language synthesis (if credentials set)

---

*Built for IBM Hackathon 2024 — Problem Statement #27 | IRIS*
