# 10 Likely Judge Questions — IRIS: Interference Recognition & Intelligence System

---

## Q1: "Why use an autoencoder for anomaly detection instead of just training a classifier on all 5 classes?"

**Short answer:**
The autoencoder is trained **only on clean signals**, so it learns the manifold of "normal". It doesn't need interference labels to flag anomalies — it flags anything it can't reconstruct well. This is **one-class / novelty detection**: it can detect interference types it has *never seen* during training (zero-shot anomaly detection). A supervised 5-class classifier needs labelled examples of every interference type and cannot handle new, unknown types. In a real deployment, new jammers appear constantly; the autoencoder remains useful even then.

---

## Q2: "Why GBM on hand-crafted features AND a CNN on spectrograms? Isn't one enough?"

**Short answer:**
Each model exploits a different signal representation. The **GBM on 12 scalar features** is fast, interpretable (feature importance ranking), and captures aggregate statistics like spectral flatness, PAPR, and duty cycle. The **CNN on spectrograms** captures *spatial patterns*: a CW tone is literally a vertical line in the spectrogram image; pulsed interference is horizontal stripes. Neither alone is always better — for ambiguous cases they disagree and the system calls for a re-check. Running both and ensemble-averaging gives higher confidence and a self-verification trigger.

---

## Q3: "What is spectral flatness and why does it distinguish interference types?"

**Short answer:**
Spectral flatness (Wiener entropy) is the ratio of the **geometric mean to the arithmetic mean** of PSD bins. It's 0 for a perfectly tonal signal (all energy in one bin), 1 for white noise (all bins equal). Clean OFDM sits at ~0.3–0.5 (multi-carrier but not uniform). CW interference pulls it toward 0. Wideband noise pushes it toward 1. Adjacent-channel leakage raises one side of the band asymmetrically, visible as PSD skewness and band-power-ratio. These metrics are used in professional spectrum analyzers (Rohde & Schwarz, Agilent VSA) for the same reason.

---

## Q4: "Your data is synthetic — how would this work on real RF data?"

**Short answer:**
The feature engineering maps directly to real measurements: `scipy.signal.welch` works identically on IQ samples from an RTL-SDR, USRP, or HackRF. Real validation steps would be: (1) collect labelled captures with a known interferer (e.g., turn on a microwave oven, aim an RTL-SDR at it on 2.4 GHz); (2) run `features.py` on the captures; (3) fine-tune or retrain the GBM and CNN — the architecture doesn't change; (4) compare anomaly detector threshold against real-world false positive rate. The README documents this explicitly because we wanted to be transparent for the judges.

---

## Q5: "How does the multi-agent system differ from just calling a chain of functions?"

**Short answer:**
Four key agentic properties: **(1) Autonomy** — the monitor agent triggers the full pipeline from an anomaly score alone with no user input. **(2) Planning** — the Orchestrator uses Granite to plan tool call order, not a hardcoded chain. **(3) Self-verification** — if classifier confidence is below 65% or GBM/CNN disagree, the Diagnosis Agent re-runs analysis before concluding. A chain would blindly pass the low-confidence result downstream. **(4) Closed-loop verification** — the Mitigation Agent applies the fix AND re-runs detection, measuring actual before/after SNR. A function chain would just apply the fix and stop. Memory across incidents is also not a function-chain property.

---

## Q6: "Why IBM Granite-13b-chat specifically? Why not GPT-4 or Llama?"

**Short answer:**
**Hackathon constraint: IBM stack required**. Granite-13b-chat is IBM's enterprise-grade instruction-tuned model with: (1) lower hallucination rate on domain-specific tasks when RAG context is provided; (2) IBM watsonx.ai platform gives reproducible, governed inference with full audit trail — important for RF/telecom deployments where you need to explain every recommendation; (3) `temperature=0.2` keeps outputs deterministic and measurement-grounded; (4) available on IBM Cloud with no data egress concerns. For production we'd benchmark Granite-20b or the instruct variants.

---

## Q7: "What happens when the mitigation doesn't work? What does the agent do?"

**Short answer:**
The `re_run_detection` tool computes `mitigation_effective = (anomaly_score_after < anomaly_score_before × 0.6)`. If `effective=False`, the Report Agent marks it as "PARTIAL" in the incident record. In the current version the agent escalates to the human operator via the dashboard (amber badge instead of green). In a next version, the Mitigation Agent would retry with an alternative strategy — e.g., pulse blanking fails at high duty cycle, so it would fall back to bandpass filtering. This retry loop is already designed into the `STRATEGY_MAP` in `MitigationAgent` and could be extended with an ordered fallback list.

---

## Q8: "How do you prevent the LLM from hallucinating RF measurements?"

**Short answer:**
Three layers of grounding: **(1) System prompt engineering** — every agent prompt says "Answer ONLY based on the provided measurements and KB excerpts. Do NOT invent measurements." **(2) Context injection** — the actual anomaly score, classifier probabilities, SNR estimate, and feature values are passed verbatim in the user prompt. Granite summarises them; it doesn't generate them. **(3) Deterministic fallback** — every agent has a rule-based `_deterministic_*` function that runs if Granite is unavailable OR returns `__DETERMINISTIC__` sentinel. The dashboard always shows the raw tool outputs alongside the LLM narrative, so a judge can verify every claim against the numbers.

---

## Q9: "Why Streamlit and not a React/Flask frontend?"

**Short answer:**
**One-day hackathon constraint**. Streamlit with Plotly gives a production-quality data dashboard in ~500 lines of Python, fully runnable with `streamlit run dashboard.py` — no Node.js build step, no REST API layer to debug. The premium red/black theme is achieved via a custom CSS injection (Google Fonts `Inter`, custom card classes, Plotly dark colorscale). For production, we'd expose the agent_system.py as a FastAPI REST endpoint and wire it to a React frontend, but the architecture is already separated: dashboard.py only calls `get_orchestrator().run_pipeline(iq)` — swapping the front-end is a one-file change.

---

## Q10: "How would you scale this to monitor thousands of sensors in real time?"

**Short answer:**
The current architecture is designed for single-sensor demo. At scale: **(1) Signal ingestion**: each sensor streams IQ chunks to a Kafka topic; (2) `MonitorAgent.run()` is stateless and ~50ms per call — horizontally scalable as a Code Engine job consuming from Kafka; (3) Full pipeline (`DiagnosisAgent` + Granite call) only triggers on anomalies (~5–20% of samples) — fan-out on detection events; (4) ChromaDB → replace with IBM Watson Discovery or a managed pgvector for HA; (5) `incidents.json` → replace with IBM Cloudant or PostgreSQL; (6) Code Engine auto-scales the dashboard container on traffic. The IBM Cloud deployment path (Code Engine + IBM Container Registry) is already in the Dockerfile and README.

---

*Memorise the one-line answer for each. Have the dashboard open. Keep backup slides for Q3 (spectral flatness graph) and Q7 (anomaly score before/after chart).*
