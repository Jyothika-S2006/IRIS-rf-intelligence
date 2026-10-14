"""
agent_tools.py — Tool Layer for the Multi-Agent System
=======================================================
All tools called by the agents. Each tool:
  - Runs real inference (loaded models / DSP) — NEVER fabricates measurements.
  - Returns a structured dict AND appends a human-readable entry to the
    global reasoning trace so the dashboard can show agent activity.
  - Is safe to call independently (dashboard, smoke test, direct script).

Tools:
  analyze_signal(iq)          → anomaly score, is_anomaly flag, raw features
  classify_signal(iq)         → GBM + CNN predictions, confidence, agreement flag
  get_spectrum_data(iq)       → PSD / spectrogram arrays for plotting
  apply_mitigation(iq, type)  → filtered IQ after applying the chosen fix
  re_run_detection(iq)        → fresh anomaly score post-mitigation
  retrieve_kb(query)          → top-k RAG chunks from ChromaDB
  store_incident(record)      → persist to incidents.json memory
  get_incident_memory(n)      → last-n incidents from memory

Model loading is lazy (first call) so the dashboard and agents can import
this file before models are trained without crashing.
"""

import json
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DATA_DIR  = ROOT / "data"
KB_DIR    = ROOT / "kb"
MEM_FILE  = ROOT / "data" / "incidents.json"
DATA_DIR.mkdir(exist_ok=True)

# ── Shared in-memory reasoning trace (thread-safe ring buffer) ────────────────
_trace: list[dict] = []
_trace_lock = Lock()
MAX_TRACE = 200


def _log(agent: str, action: str, detail: str = "", data: Any = None):
    """Append one entry to the reasoning trace."""
    entry = {
        "ts":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "agent":  agent,
        "action": action,
        "detail": detail,
        "data":   data,
    }
    with _trace_lock:
        _trace.append(entry)
        if len(_trace) > MAX_TRACE:
            _trace.pop(0)
    return entry


def get_trace(last_n: int = 50) -> list[dict]:
    """Return the last N trace entries (used by dashboard)."""
    with _trace_lock:
        return list(_trace[-last_n:])


def clear_trace():
    with _trace_lock:
        _trace.clear()


# ── Lazy model loader ─────────────────────────────────────────────────────────
_models: dict = {}
_models_lock = Lock()

FEATURE_COLS = [
    "psd_peak_power_db", "psd_mean_power_db", "spectral_flatness",
    "occupied_bandwidth", "papr_db", "duty_cycle_est", "spectral_kurtosis",
    "peak_freq_offset", "psd_skewness", "band_power_ratio",
    "rms_amplitude", "freq_std",
]
CLASS_NAMES = ["clean", "narrowband_cw", "wideband_noise", "pulsed", "adjacent_channel"]


def _load_models():
    """Load all saved models once. Safe to call multiple times."""
    global _models
    with _models_lock:
        if _models:
            return _models

        import joblib

        # ── GBM ───────────────────────────────────────────────────────────
        gbm_path = MODEL_DIR / "gbm_classifier.pkl"
        if gbm_path.exists():
            _models["gbm"]     = joblib.load(MODEL_DIR / "gbm_classifier.pkl")
            _models["scaler_gbm"] = joblib.load(MODEL_DIR / "scaler_gbm.pkl")
            _models["le"]      = joblib.load(MODEL_DIR / "label_encoder.pkl")
        else:
            _models["gbm"] = None

        # ── Autoencoder ───────────────────────────────────────────────────
        ae_path = MODEL_DIR / "autoencoder.pt"
        _models["ae"] = None
        if ae_path.exists() and ae_path.stat().st_size > 1024:
            try:
                import torch
                import torch.nn as nn

                class Autoencoder(nn.Module):
                    def __init__(self, n_in):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(n_in, 64), nn.ReLU(),
                            nn.Linear(64, 32),   nn.ReLU(),
                            nn.Linear(32, 8),    nn.ReLU(),
                        )
                        self.decoder = nn.Sequential(
                            nn.Linear(8, 32),   nn.ReLU(),
                            nn.Linear(32, 64),  nn.ReLU(),
                            nn.Linear(64, n_in),
                        )
                    def forward(self, x):
                        return self.decoder(self.encoder(x))

                ae_cfg  = json.load(open(MODEL_DIR / "ae_threshold.json"))
                ae_mdl  = Autoencoder(ae_cfg["n_in"])
                ae_mdl.load_state_dict(
                    torch.load(ae_path, map_location="cpu", weights_only=True))
                ae_mdl.eval()
                _models["ae"]           = ae_mdl
                _models["ae_threshold"] = ae_cfg["threshold"]
                _models["scaler_ae"]    = joblib.load(MODEL_DIR / "scaler_ae.pkl")
            except Exception as e:
                print(f"  [warn] Autoencoder not loaded: {e}")
                _models["ae"] = None

        # ── CNN ───────────────────────────────────────────────────────────
        cnn_path = MODEL_DIR / "cnn_classifier.pt"
        _models["cnn"] = None
        if cnn_path.exists() and cnn_path.stat().st_size > 1024:
            try:
                import torch
                import torch.nn as nn

                class SmallCNN(nn.Module):
                    def __init__(self, n_classes):
                        super().__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(4),
                        )
                        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, n_classes),
                        )
                    def forward(self, x):
                        return self.classifier(self.features(x))

                cnn_cfg = json.load(open(MODEL_DIR / "cnn_classes.json"))
                cnn_mdl = SmallCNN(cnn_cfg["n_classes"])
                cnn_mdl.load_state_dict(
                    torch.load(cnn_path, map_location="cpu", weights_only=True))
                cnn_mdl.eval()
                _models["cnn"]         = cnn_mdl
                _models["cnn_classes"] = cnn_cfg["classes"]
            except Exception as e:
                print(f"  [warn] CNN not loaded: {e}")
                _models["cnn"] = None

        return _models


def models_available() -> dict:
    """Return which models are loaded (for dashboard status indicator)."""
    m = _load_models()
    return {
        "gbm": m.get("gbm") is not None,
        "autoencoder": m.get("ae") is not None,
        "cnn": m.get("cnn") is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: analyze_signal
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_signal(iq: np.ndarray, sample_id: str = "live") -> dict:
    """
    Run the autoencoder anomaly detector on a raw IQ array.

    Returns
    -------
    {
      "sample_id": str,
      "anomaly_score": float,      # reconstruction MSE
      "threshold": float,
      "is_anomaly": bool,
      "features": dict,            # 12 scalar features
      "snr_estimate_db": float,
    }
    """
    from features import extract_features_from_iq, extract_features

    m = _load_models()
    feat_vec = extract_features_from_iq(iq)
    feat_dict = extract_features(iq)

    # Estimate SNR from peak vs median PSD ratio
    from features import compute_welch_psd
    _, pxx = compute_welch_psd(iq)
    snr_est = float(10 * np.log10(pxx.max() / (np.median(pxx) + 1e-20)))

    if m["ae"] is None:
        _log("ToolLayer", "analyze_signal", "Autoencoder not loaded - returning stub")
        return {
            "sample_id": sample_id, "anomaly_score": 0.0,
            "threshold": 1.0, "is_anomaly": False,
            "features": feat_dict, "snr_estimate_db": snr_est,
            "error": "model_not_loaded",
        }

    import torch
    scaler = m["scaler_ae"]
    X_s    = scaler.transform(feat_vec.reshape(1, -1))
    x_t    = torch.FloatTensor(X_s)
    with torch.no_grad():
        recon = m["ae"](x_t).numpy()
    score = float(np.mean((recon - X_s) ** 2))
    thr   = float(m["ae_threshold"])
    is_an = bool(score > thr)

    result = {
        "sample_id":       sample_id,
        "anomaly_score":   round(score, 6),
        "threshold":       round(thr, 6),
        "is_anomaly":      is_an,
        "features":        {k: round(float(v), 4) for k, v in feat_dict.items()},
        "snr_estimate_db": round(snr_est, 2),
    }
    _log("MonitorAgent", "analyze_signal",
         f"score={score:.5f} thr={thr:.5f} anomaly={is_an}", result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: classify_signal
# ═══════════════════════════════════════════════════════════════════════════════

def classify_signal(iq: np.ndarray, sample_id: str = "live") -> dict:
    """
    Run GBM + CNN classifiers. If they disagree AND confidence < threshold,
    sets needs_recheck=True so the Diagnosis Agent knows to re-analyse.

    Returns
    -------
    {
      "gbm_class": str, "gbm_confidence": float, "gbm_probs": dict,
      "cnn_class": str, "cnn_confidence": float,   (if CNN available)
      "final_class": str, "final_confidence": float,
      "agreement": bool,
      "needs_recheck": bool,  # True if confidence < 0.65 or models disagree
    }
    """
    from features import extract_features_from_iq, compute_stft_spectrogram

    m = _load_models()
    feat_vec = extract_features_from_iq(iq)

    result: dict = {}
    CONF_THRESHOLD = 0.65

    # ── GBM ──────────────────────────────────────────────────────────────
    if m["gbm"] is not None:
        X_s   = m["scaler_gbm"].transform(feat_vec.reshape(1, -1))
        probs = m["gbm"].predict_proba(X_s)[0]
        classes = m["le"].classes_
        gbm_idx  = int(np.argmax(probs))
        raw_cls  = classes[gbm_idx]
        # LabelEncoder may store int labels (0-4) or string names — normalise to string
        try:
            gbm_cls = CLASS_NAMES[int(raw_cls)]
        except (ValueError, IndexError, KeyError):
            gbm_cls = str(raw_cls)
        gbm_conf = float(probs[gbm_idx])
        result["gbm_class"]      = gbm_cls
        result["gbm_confidence"] = round(gbm_conf, 4)
        result["gbm_probs"]      = {
            CLASS_NAMES[int(c)] if str(c).isdigit() else str(c): round(float(p), 4)
            for c, p in zip(classes, probs)
        }
    else:
        result["gbm_class"] = "unknown"; result["gbm_confidence"] = 0.0
        result["gbm_probs"] = {}

    # ── CNN ───────────────────────────────────────────────────────────────
    if m["cnn"] is not None:
        import torch
        import torch.nn.functional as F
        spec = compute_stft_spectrogram(iq).astype(np.float32) / 255.0
        x_t  = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
        with torch.no_grad():
            logits = m["cnn"](x_t)
            probs_c = F.softmax(logits, dim=1).numpy()[0]
        cnn_idx  = int(np.argmax(probs_c))
        cnn_cls  = m["cnn_classes"][cnn_idx]
        cnn_conf = float(probs_c[cnn_idx])
        result["cnn_class"]      = cnn_cls
        result["cnn_confidence"] = round(cnn_conf, 4)
    else:
        cnn_cls  = result["gbm_class"]
        cnn_conf = result["gbm_confidence"]
        result["cnn_class"] = "unavailable"; result["cnn_confidence"] = 0.0

    # ── Agreement & final decision ────────────────────────────────────────
    gbm_cls  = result["gbm_class"]
    gbm_conf = result["gbm_confidence"]
    agreement = (gbm_cls == cnn_cls)

    # Ensemble: if both available and agree, boost confidence; else GBM primary
    if m["cnn"] is not None and agreement:
        final_cls  = gbm_cls
        final_conf = round(min(0.99, (gbm_conf + cnn_conf) / 2 + 0.05), 4)
    else:
        final_cls  = gbm_cls   # GBM is primary
        final_conf = gbm_conf

    needs_recheck = (final_conf < CONF_THRESHOLD) or (not agreement and m["cnn"] is not None)

    result["final_class"]      = final_cls
    result["final_confidence"] = final_conf
    result["agreement"]        = agreement
    result["needs_recheck"]    = needs_recheck

    _log("DiagnosisAgent", "classify_signal",
         f"GBM={gbm_cls}({gbm_conf:.2f}) CNN={cnn_cls}({cnn_conf:.2f}) "
         f"agree={agreement} recheck={needs_recheck}", result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: get_spectrum_data
# ═══════════════════════════════════════════════════════════════════════════════

def get_spectrum_data(iq: np.ndarray) -> dict:
    """
    Return PSD and spectrogram arrays for dashboard plotting (no file I/O).
    """
    from features import compute_welch_psd, compute_stft_spectrogram
    f, pxx = compute_welch_psd(iq)
    pxx_db = 10 * np.log10(pxx + 1e-20)
    spec   = compute_stft_spectrogram(iq)
    return {
        "freq_norm":  f.tolist(),
        "psd_db":     pxx_db.tolist(),
        "spectrogram": spec.tolist(),   # list of lists (64×64 uint8)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: apply_mitigation
# ═══════════════════════════════════════════════════════════════════════════════

def apply_mitigation(iq: np.ndarray, interference_type: str,
                     features: dict = None) -> dict:
    """
    Apply the appropriate DSP mitigation to the IQ signal.

    Supported strategies:
      narrowband_cw      → IIR notch filter at peak frequency
      adjacent_channel   → bandpass filter (trim asymmetric shoulders)
      wideband_noise     → bandpass filter (restrict bandwidth)
      pulsed             → pulse blanking (zero samples above power threshold)
      clean              → no-op (pass-through)

    Returns: {"filtered_iq": np.ndarray, "strategy": str, "params": dict}
    """
    from scipy.signal import iirnotch, butter, filtfilt, sosfilt, butter

    FS = 1.0   # normalised (freq is fraction of FS)
    strategy = "none"
    params   = {}

    iq_out = iq.copy()

    if interference_type == "narrowband_cw":
        # Notch at detected peak frequency
        peak_freq = float(features.get("peak_freq_offset", 0.0)) if features else 0.0
        f0 = abs(peak_freq)  # normalised 0..0.5
        if f0 < 0.01:
            f0 = 0.05        # fallback if near DC
        Q  = 35.0
        # iirnotch works on real signal; apply to real & imag separately
        b, a = iirnotch(f0 * 2, Q)  # w0 in [0,1] where 1=Nyquist → *2 for normalised
        iq_out = (filtfilt(b, a, iq.real) + 1j * filtfilt(b, a, iq.imag)).astype(np.complex64)
        strategy = "notch_filter"
        params   = {"notch_freq_norm": round(peak_freq, 4), "Q": Q}

    elif interference_type in ("adjacent_channel",):
        # Bandpass: keep centre 80% of the band
        low, high = 0.1, 0.9  # normalised Nyquist fractions
        sos = butter(5, [low, high], btype="band", output="sos")
        iq_out = (sosfilt(sos, iq.real) + 1j * sosfilt(sos, iq.imag)).astype(np.complex64)
        strategy = "bandpass_filter"
        params   = {"low_norm": low, "high_norm": high, "order": 5}

    elif interference_type == "wideband_noise":
        # Bandpass: restrict to 60% of band
        low, high = 0.2, 0.8
        sos = butter(5, [low, high], btype="band", output="sos")
        iq_out = (sosfilt(sos, iq.real) + 1j * sosfilt(sos, iq.imag)).astype(np.complex64)
        strategy = "bandpass_filter_narrow"
        params   = {"low_norm": low, "high_norm": high, "order": 5}

    elif interference_type == "pulsed":
        # Pulse blanking: zero samples above 90th-percentile power
        env = np.abs(iq) ** 2
        thr = float(np.percentile(env, 90))
        mask = env > thr
        iq_out = iq.copy()
        iq_out[mask] = 0.0 + 0.0j
        strategy = "pulse_blanking"
        params   = {"blanked_fraction": round(float(mask.mean()), 4),
                    "power_threshold":  round(float(thr), 6)}

    else:
        strategy = "no_mitigation"

    _log("MitigationAgent", "apply_mitigation",
         f"type={interference_type} → strategy={strategy}", params)
    return {
        "filtered_iq": iq_out.astype(np.complex64),
        "strategy":    strategy,
        "params":      params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: re_run_detection
# ═══════════════════════════════════════════════════════════════════════════════

def re_run_detection(iq_before: np.ndarray, iq_after: np.ndarray,
                     sample_id: str = "live") -> dict:
    """
    Re-run anomaly detection on the mitigated signal and compute before/after SNR.
    The Mitigation Agent calls this to close the loop and verify the fix worked.

    Returns
    -------
    {
      "before": {anomaly_score, is_anomaly, snr_estimate_db},
      "after":  {anomaly_score, is_anomaly, snr_estimate_db},
      "snr_improvement_db": float,
      "mitigation_effective": bool,
    }
    """
    before = analyze_signal(iq_before, sample_id + "_before")
    after  = analyze_signal(iq_after,  sample_id + "_after")

    snr_delta = round(
        after["snr_estimate_db"] - before["snr_estimate_db"], 2)
    effective = (not after["is_anomaly"]) or (
        after["anomaly_score"] < before["anomaly_score"] * 0.6)

    result = {
        "before":                before,
        "after":                 after,
        "snr_improvement_db":    snr_delta,
        "anomaly_score_delta":   round(after["anomaly_score"] - before["anomaly_score"], 6),
        "mitigation_effective":  effective,
    }
    _log("MitigationAgent", "re_run_detection",
         f"SNR Δ={snr_delta:+.2f} dB  effective={effective}", result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6: retrieve_kb
# ═══════════════════════════════════════════════════════════════════════════════

# ChromaDB vector store — built once then reused
_chroma_collection = None
_chroma_lock = Lock()


def _build_chroma():
    """
    Build (or load) the ChromaDB vector store from the KB markdown files.
    Uses sentence-transformers all-MiniLM-L6-v2 (small, CPU-friendly, ~80 MB).
    Falls back to TF-IDF keyword search if chromadb/sentence-transformers not installed.
    """
    global _chroma_collection
    with _chroma_lock:
        if _chroma_collection is not None:
            return _chroma_collection

        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction)

            ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2")
            client = chromadb.Client()
            col    = client.get_or_create_collection(
                "rf_kb", embedding_function=ef)

            if col.count() == 0:
                _ingest_kb_to_chroma(col)

            _chroma_collection = col
        except ImportError:
            _chroma_collection = "tfidf"   # fallback sentinel

    return _chroma_collection


def _ingest_kb_to_chroma(col):
    """Split KB markdown files into chunks and upsert into ChromaDB."""
    docs, ids, metas = [], [], []
    chunk_size = 400   # characters per chunk
    idx = 0
    for md_file in sorted(KB_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Split on double newlines (paragraphs)
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        for para in paras:
            # Further chunk if very long
            for i in range(0, len(para), chunk_size):
                chunk = para[i:i + chunk_size]
                if len(chunk.strip()) < 30:
                    continue
                docs.append(chunk)
                ids.append(f"kb_{idx}")
                metas.append({"source": md_file.name})
                idx += 1
    col.upsert(documents=docs, ids=ids, metadatas=metas)
    print(f"  ChromaDB: ingested {idx} chunks from {KB_DIR}")


def _tfidf_search(query: str, top_k: int = 3) -> list[dict]:
    """Simple keyword fallback when ChromaDB is not available."""
    results = []
    query_words = set(query.lower().split())
    for md_file in sorted(KB_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        for para in paras:
            words = set(para.lower().split())
            score = len(query_words & words) / (len(query_words) + 1)
            results.append({"text": para[:400], "source": md_file.name, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def retrieve_kb(query: str, top_k: int = 3) -> dict:
    """
    Semantic search over the KB.  Returns grounding documents for the Diagnosis
    and Mitigation agents so answers are always sourced, never invented.
    """
    col = _build_chroma()
    if col == "tfidf":
        chunks = _tfidf_search(query, top_k)
        _log("DiagnosisAgent", "retrieve_kb",
             f"TF-IDF fallback | query={query!r}", chunks)
        return {"query": query, "results": chunks, "backend": "tfidf"}

    res = col.query(query_texts=[query], n_results=top_k)
    chunks = [
        {"text": doc, "source": meta.get("source", "kb"), "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    _log("DiagnosisAgent", "retrieve_kb",
         f"ChromaDB | query={query!r} | {len(chunks)} results", chunks)
    return {"query": query, "results": chunks, "backend": "chromadb"}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7 & 8: Persistent incident memory
# ═══════════════════════════════════════════════════════════════════════════════

_mem_lock = Lock()


def store_incident(record: dict) -> dict:
    """
    Persist an incident record to incidents.json.
    record must contain: type, confidence, mitigation, snr_before, snr_after, effective.
    """
    with _mem_lock:
        if MEM_FILE.exists():
            incidents = json.loads(MEM_FILE.read_text())
        else:
            incidents = []
        record["incident_id"] = str(uuid.uuid4())[:8]
        record["timestamp"]   = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        incidents.append(record)
        MEM_FILE.write_text(json.dumps(incidents, indent=2))
    _log("ReportAgent", "store_incident",
         f"#{record['incident_id']} {record.get('type','?')} saved", record)
    return {"status": "ok", "incident_id": record["incident_id"]}


def get_incident_memory(n: int = 5) -> list[dict]:
    """Return the last n incident records from memory."""
    with _mem_lock:
        if not MEM_FILE.exists():
            return []
        incidents = json.loads(MEM_FILE.read_text())
    recent = incidents[-n:]
    _log("OrchestratorAgent", "get_incident_memory",
         f"Retrieved {len(recent)} past incidents")
    return recent


# ═══════════════════════════════════════════════════════════════════════════════
# Quick self-test (no models needed)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    iq  = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
    print("=== agent_tools.py self-test ===")
    print("KB retrieval (tfidf fallback):")
    r = retrieve_kb("notch filter narrowband CW interference")
    for chunk in r["results"]:
        print(f"  [{chunk['source']}] {chunk['text'][:80]}…")

    print("\nSpectrum data:")
    sd = get_spectrum_data(iq)
    print(f"  freq bins={len(sd['freq_norm'])}  psd bins={len(sd['psd_db'])}  "
          f"spec shape={len(sd['spectrogram'])}×{len(sd['spectrogram'][0])}")

    avail = models_available()
    print(f"\nModels available: {avail}")
    if avail["gbm"]:
        r2 = classify_signal(iq)
        print(f"  Classify: {r2['final_class']} ({r2['final_confidence']:.2f})")
    else:
        print("  (Models not loaded — run train_all.py first)")

    print("\nTrace:")
    for entry in get_trace():
        print(f"  [{entry['ts']}] {entry['agent']} / {entry['action']}: {entry['detail']}")
    print("OK")
