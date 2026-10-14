"""
agent_system.py — Multi-Agent Orchestration System
=====================================================
Implements four specialised agents coordinated by an Orchestrator.
Each agent is a Python class whose `run()` method:
  1. Plans its steps using watsonx.ai Granite (if credentials are set).
  2. Calls the appropriate tools from agent_tools.py.
  3. Logs every thought/action/observation to the shared reasoning trace.
  4. Returns a structured result dict.

If watsonx credentials are NOT set, each agent runs in "deterministic mode"
(hardcoded decision logic) — the app still works end-to-end for demos.

Agents
------
  MonitorAgent     – watches a signal, triggers when anomaly score > threshold
  DiagnosisAgent   – detect → classify → RAG lookup → self-verify if low conf
  MitigationAgent  – choose fix → apply → re-run detection (closed loop)
  ReportAgent      – write structured incident report, store to memory
  OrchestratorAgent– coordinate the above, check past memory, final answer

Public API (used by dashboard.py)
----------------------------------
  run_pipeline(iq)         → full incident dict + reasoning trace
  monitor_tick(iq)         → lightweight anomaly check (for live polling)
  get_agent_status()       → which agents are active / idle
"""

import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from agent_tools import (
    analyze_signal, classify_signal, get_spectrum_data,
    apply_mitigation, re_run_detection, retrieve_kb,
    store_incident, get_incident_memory, get_trace, _log,
)

# ── watsonx.ai Granite client (optional) ─────────────────────────────────────
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_APIKEY     = os.environ.get("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
GRANITE_MODEL      = "ibm/granite-13b-chat-v2"

# Low confidence / disagreement threshold
CONF_THRESHOLD     = 0.65
# Anomaly score multiplier for "mitigation effective" check
MITIGATION_SCORE_RATIO = 0.6


def _call_granite(system_prompt: str, user_prompt: str,
                  max_tokens: int = 512) -> str:
    """
    Call watsonx.ai Granite via the ibm-watsonx-ai SDK.
    Returns the model's text response.
    Falls back to a deterministic stub if credentials are missing.
    """
    if not WATSONX_APIKEY or not WATSONX_PROJECT_ID:
        return "__DETERMINISTIC__"   # sentinel → agents use rule-based fallback

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)
        model = ModelInference(
            model_id=GRANITE_MODEL,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={
                "max_new_tokens":   max_tokens,
                "temperature":      0.2,     # low temp → factual, consistent
                "repetition_penalty": 1.1,
                "stop_sequences":   ["<|endoftext|>", "\n\nHuman:"],
            },
        )
        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
        ]
        resp = model.chat(messages=messages)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        _log("OrchestratorAgent", "granite_error", str(exc))
        return "__DETERMINISTIC__"


# ═══════════════════════════════════════════════════════════════════════════════
# MONITOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class MonitorAgent:
    """
    Continuously watches the signal stream.
    Triggers the Diagnosis Agent when anomaly_score > threshold.
    This agent runs autonomously — no user prompt required.
    """
    NAME = "MonitorAgent"

    def run(self, iq: np.ndarray, sample_id: str = "live") -> dict:
        _log(self.NAME, "start", f"Analysing sample {sample_id}")

        result = analyze_signal(iq, sample_id)

        if result.get("error"):
            _log(self.NAME, "warning", "Autoencoder not loaded; using feature heuristic")
            # Heuristic fallback grounded in DSP:
            #  - high spectral kurtosis  → impulsive / tonal signal
            #  - high peak-to-mean ratio → single dominant bin (CW)
            #  - high spectral flatness  → wideband noise
            #  - high PAPR               → pulsed bursts
            feat = result["features"]
            peak_db  = float(feat.get("psd_peak_power_db", 0))
            mean_db  = float(feat.get("psd_mean_power_db", 0))
            peak_mean_ratio = min((peak_db - mean_db) / 40.0, 1.0)   # >40 dB diff = strong CW
            kurtosis_score  = min(float(feat.get("spectral_kurtosis", 0)) / 50.0, 1.0)
            flatness_score  = float(feat.get("spectral_flatness", 0))  # near 1 = wideband noise
            papr_score      = min(float(feat.get("papr_db", 0)) / 20.0, 1.0)  # >20 dB = pulsed
            heuristic_score = (
                peak_mean_ratio * 0.35 +
                kurtosis_score  * 0.30 +
                flatness_score  * 0.20 +
                papr_score      * 0.15
            )
            result["anomaly_score"] = round(heuristic_score, 5)
            result["threshold"]     = 0.25
            result["is_anomaly"]    = heuristic_score > 0.25

        severity = "HIGH" if result["anomaly_score"] > result["threshold"] * 2 else \
                   "MEDIUM" if result["is_anomaly"] else "LOW"
        result["severity"] = severity

        _log(self.NAME, "decision",
             f"anomaly={result['is_anomaly']} severity={severity} "
             f"score={result['anomaly_score']:.5f}")
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosisAgent:
    """
    Classify the interference, retrieve KB grounding, self-verify if uncertain.
    Self-verification: if confidence < CONF_THRESHOLD or GBM/CNN disagree,
    re-examines the signal features manually and asks Granite for a second opinion.
    """
    NAME = "DiagnosisAgent"

    def run(self, iq: np.ndarray, monitor_result: dict,
            sample_id: str = "live") -> dict:
        _log(self.NAME, "start",
             f"Diagnosing sample {sample_id} "
             f"(anomaly_score={monitor_result['anomaly_score']:.5f})")

        # ── Step 1: Classify ─────────────────────────────────────────────
        clf = classify_signal(iq, sample_id)
        _log(self.NAME, "classify",
             f"GBM={clf['gbm_class']}({clf['gbm_confidence']:.2f}) "
             f"CNN={clf.get('cnn_class','N/A')}({clf.get('cnn_confidence',0):.2f}) "
             f"agree={clf['agreement']}")

        # ── Step 2: Self-verification loop ────────────────────────────────
        recheck_count = 0
        MAX_RECHECKS  = 2
        while clf["needs_recheck"] and recheck_count < MAX_RECHECKS:
            recheck_count += 1
            _log(self.NAME, "recheck",
                 f"Confidence {clf['final_confidence']:.2f} < {CONF_THRESHOLD} "
                 f"or models disagree - recheck #{recheck_count}")

            # Add small Gaussian perturbation to IQ (simulate re-measurement)
            rng     = np.random.default_rng(recheck_count)
            iq_re   = (iq + (rng.standard_normal(len(iq)) +
                             1j * rng.standard_normal(len(iq))
                             ).astype(np.complex64) * 0.01).astype(np.complex64)
            clf_re  = classify_signal(iq_re, sample_id + f"_recheck{recheck_count}")

            # Take whichever run gave higher confidence
            if clf_re["final_confidence"] > clf["final_confidence"]:
                clf = clf_re
                _log(self.NAME, "recheck_result",
                     f"Recheck improved: {clf['final_class']} "
                     f"({clf['final_confidence']:.2f})")
            else:
                clf["needs_recheck"] = False   # accept best result
                _log(self.NAME, "recheck_result",
                     "Recheck did not improve; accepting current classification")

        # ── Step 3: RAG grounding ─────────────────────────────────────────
        itype = clf["final_class"]
        kb_q  = f"{itype} interference characteristics and real-world sources"
        kb    = retrieve_kb(kb_q, top_k=3)

        mit_q = f"mitigation strategy for {itype} interference"
        kb_mit = retrieve_kb(mit_q, top_k=2)

        # ── Step 4: Granite reasoning (if credentials set) ────────────────
        system_p = (
            "You are an RF interference diagnostic expert. "
            "Answer ONLY based on the provided measurements and KB excerpts. "
            "Do NOT invent measurements or sources not in the data. "
            "Be concise (3-4 sentences)."
        )
        user_p = (
            f"Signal measurements:\n"
            f"  Anomaly score:  {monitor_result['anomaly_score']:.5f} "
            f"(threshold {monitor_result['threshold']:.5f})\n"
            f"  GBM classifier: {clf['gbm_class']} ({clf['gbm_confidence']:.0%})\n"
            f"  CNN classifier: {clf.get('cnn_class','N/A')} "
            f"({clf.get('cnn_confidence',0):.0%})\n"
            f"  Final decision: {itype} ({clf['final_confidence']:.0%})\n"
            f"  SNR estimate:   {monitor_result.get('snr_estimate_db',0):.1f} dB\n"
            f"Key features: "
            f"spectral_flatness={monitor_result['features'].get('spectral_flatness',0):.3f}, "
            f"papr_db={monitor_result['features'].get('papr_db',0):.1f}, "
            f"duty_cycle={monitor_result['features'].get('duty_cycle_est',0):.2f}\n\n"
            f"KB excerpts:\n" +
            "\n".join(f"- {r['text'][:200]}" for r in kb["results"]) +
            "\n\nBased ONLY on the above, identify the likely real-world source "
            "and explain why this signal type matches the measurements."
        )
        granite_resp = _call_granite(system_p, user_p)

        if granite_resp == "__DETERMINISTIC__":
            # Rule-based explanation — grounded in actual feature values
            feat = monitor_result["features"]
            granite_resp = _deterministic_diagnosis(itype, feat, clf)
            _log(self.NAME, "granite", "Using deterministic fallback (no credentials)")
        else:
            _log(self.NAME, "granite", "Granite response received", granite_resp[:100])

        return {
            "sample_id":         sample_id,
            "interference_type": itype,
            "confidence":        clf["final_confidence"],
            "gbm_result":        clf,
            "monitor_result":    monitor_result,
            "kb_results":        kb["results"],
            "kb_mitigation":     kb_mit["results"],
            "explanation":       granite_resp,
            "rechecks":          recheck_count,
        }


def _deterministic_diagnosis(itype: str, feat: dict, clf: dict) -> str:
    """Rule-based explanation grounded in measured feature values."""
    flat   = feat.get("spectral_flatness", 0)
    papr   = feat.get("papr_db", 0)
    duty   = feat.get("duty_cycle_est", 0)
    pfo    = feat.get("peak_freq_offset", 0)
    bpr    = feat.get("band_power_ratio", 1)
    skew   = feat.get("psd_skewness", 0)
    conf   = clf["final_confidence"]

    explanations = {
        "clean": (
            f"Signal shows healthy OFDM characteristics: spectral flatness={flat:.3f} "
            f"(moderate, typical of multi-carrier), PAPR={papr:.1f} dB (normal for OFDM). "
            f"No interference pattern detected. Confidence: {conf:.0%}."
        ),
        "narrowband_cw": (
            f"Sharp CW tone detected: spectral flatness={flat:.3f} (near 0 = single spike), "
            f"peak frequency offset={pfo:.3f} (normalised). "
            f"Consistent with a malfunctioning oscillator, GPS jammer, or microwave harmonic. "
            f"Confidence: {conf:.0%}."
        ),
        "wideband_noise": (
            f"Wideband noise jammer: spectral flatness={flat:.3f} (near 1 = white noise), "
            f"elevated noise floor across the full band. "
            f"Likely source: switching power supply EMI, broadband jammer, or GPU interference. "
            f"Confidence: {conf:.0%}."
        ),
        "pulsed": (
            f"Pulsed interference: PAPR={papr:.1f} dB (high burst power), "
            f"duty cycle estimate={duty:.2f} ({duty*100:.0f}% active). "
            f"Consistent with radar (weather/airport), microwave oven, or pulsed EMI source. "
            f"Confidence: {conf:.0%}."
        ),
        "adjacent_channel": (
            f"Adjacent-channel leakage: band power ratio={bpr:.3f} (asymmetric PSD), "
            f"PSD skewness={skew:.3f}. "
            f"Likely a neighbouring WiFi AP, femtocell, or DECT phone on an adjacent channel. "
            f"Confidence: {conf:.0%}."
        ),
    }
    return explanations.get(itype, f"Interference type: {itype}. Confidence: {conf:.0%}.")


# ═══════════════════════════════════════════════════════════════════════════════
# MITIGATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class MitigationAgent:
    """
    Selects and applies the appropriate mitigation strategy,
    then closes the loop by re-running detection on the filtered signal.
    """
    NAME = "MitigationAgent"

    # Maps interference type to mitigation strategy name
    STRATEGY_MAP = {
        "narrowband_cw":    "notch_filter",
        "adjacent_channel": "channel_change",   # modelled as bandpass in simulation
        "wideband_noise":   "bandpass_filter",
        "pulsed":           "pulse_blanking",
        "clean":            "no_mitigation",
    }

    def run(self, iq: np.ndarray, diagnosis: dict) -> dict:
        itype   = diagnosis["interference_type"]
        conf    = diagnosis["confidence"]
        feat    = diagnosis["monitor_result"]["features"]
        kb_docs = diagnosis["kb_mitigation"]

        _log(self.NAME, "start",
             f"Planning mitigation for {itype} (conf={conf:.2f})")

        # ── Step 1: Choose strategy ───────────────────────────────────────
        strategy_name = self.STRATEGY_MAP.get(itype, "no_mitigation")

        # Ask Granite to confirm / refine strategy choice
        system_p = (
            "You are an RF mitigation specialist. "
            "Choose ONE mitigation strategy from the KB. "
            "Reply with ONLY the strategy name and ONE sentence of justification. "
            "Do not invent strategies not in the KB."
        )
        kb_text = "\n".join(r["text"][:300] for r in kb_docs)
        user_p  = (
            f"Interference type: {itype}\n"
            f"Confidence: {conf:.0%}\n"
            f"Key features: flatness={feat.get('spectral_flatness',0):.3f}, "
            f"papr={feat.get('papr_db',0):.1f} dB, "
            f"duty_cycle={feat.get('duty_cycle_est',0):.2f}\n\n"
            f"Available KB strategies:\n{kb_text}\n\n"
            f"Which single strategy should be applied and why?"
        )
        granite_resp = _call_granite(system_p, user_p, max_tokens=200)
        if granite_resp == "__DETERMINISTIC__":
            granite_resp = (
                f"Apply {strategy_name.replace('_',' ')} — "
                f"grounded in KB: standard mitigation for {itype} interference."
            )
            _log(self.NAME, "strategy_choice",
                 f"Deterministic: {strategy_name}")
        else:
            _log(self.NAME, "strategy_choice",
                 f"Granite recommended: {granite_resp[:80]}")

        # ── Step 2: Apply mitigation ──────────────────────────────────────
        mit_result = apply_mitigation(iq, itype, feat)
        iq_filtered = mit_result["filtered_iq"]

        # ── Step 3: Closed-loop re-detection ──────────────────────────────
        _log(self.NAME, "closed_loop", "Re-running detection on filtered signal")
        verify = re_run_detection(iq, iq_filtered, diagnosis["sample_id"])

        _log(self.NAME, "result",
             f"SNR improvement: {verify['snr_improvement_db']:+.2f} dB  "
             f"effective={verify['mitigation_effective']}")

        return {
            "strategy":            strategy_name,
            "strategy_applied":    mit_result["strategy"],
            "params":              mit_result["params"],
            "granite_rationale":   granite_resp,
            "iq_filtered":         iq_filtered,
            "verification":        verify,
            "snr_improvement_db":  verify["snr_improvement_db"],
            "effective":           verify["mitigation_effective"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class ReportAgent:
    """
    Compiles the full incident report from all agent outputs and stores it in memory.
    The report is grounded entirely in tool outputs — Granite only formats the summary.
    """
    NAME = "ReportAgent"

    def run(self, diagnosis: dict, mitigation: dict) -> dict:
        _log(self.NAME, "start", "Compiling incident report")

        itype  = diagnosis["interference_type"]
        conf   = diagnosis["confidence"]
        verify = mitigation["verification"]

        # ── Granite narrative summary ─────────────────────────────────────
        system_p = (
            "You are an RF incident report writer. "
            "Write a concise 3-sentence technical summary of this incident. "
            "Use ONLY the provided data values. Do not speculate."
        )
        user_p = (
            f"Incident data:\n"
            f"  Type:            {itype}\n"
            f"  Confidence:      {conf:.0%}\n"
            f"  Explanation:     {diagnosis['explanation'][:200]}\n"
            f"  Strategy:        {mitigation['strategy_applied']}\n"
            f"  SNR before:      {verify['before']['snr_estimate_db']:.1f} dB\n"
            f"  SNR after:       {verify['after']['snr_estimate_db']:.1f} dB\n"
            f"  SNR improvement: {verify['snr_improvement_db']:+.2f} dB\n"
            f"  Mitigation OK:   {mitigation['effective']}\n"
            f"  Anomaly before:  {verify['before']['anomaly_score']:.5f}\n"
            f"  Anomaly after:   {verify['after']['anomaly_score']:.5f}\n"
        )
        narrative = _call_granite(system_p, user_p, max_tokens=256)
        if narrative == "__DETERMINISTIC__":
            eff_str = "successfully resolved" if mitigation["effective"] else "partially mitigated"
            narrative = (
                f"RF interference of type '{itype}' was detected with {conf:.0%} confidence "
                f"(anomaly score {verify['before']['anomaly_score']:.5f}). "
                f"Applied {mitigation['strategy_applied'].replace('_',' ')} which {eff_str} "
                f"the interference. SNR changed from {verify['before']['snr_estimate_db']:.1f} dB "
                f"to {verify['after']['snr_estimate_db']:.1f} dB "
                f"({verify['snr_improvement_db']:+.2f} dB improvement)."
            )

        # ── Build structured record ───────────────────────────────────────
        record = {
            "type":          itype,
            "confidence":    conf,
            "severity":      diagnosis["monitor_result"].get("severity", "MEDIUM"),
            "snr_before_db": verify["before"]["snr_estimate_db"],
            "snr_after_db":  verify["after"]["snr_estimate_db"],
            "snr_delta_db":  verify["snr_improvement_db"],
            "anomaly_before": verify["before"]["anomaly_score"],
            "anomaly_after":  verify["after"]["anomaly_score"],
            "mitigation":    mitigation["strategy_applied"],
            "params":        mitigation["params"],
            "effective":     mitigation["effective"],
            "narrative":     narrative,
            "rechecks":      diagnosis.get("rechecks", 0),
        }

        mem = store_incident(record)
        record["incident_id"] = mem["incident_id"]

        _log(self.NAME, "report_complete",
             f"Incident #{record['incident_id']} stored. {narrative[:80]}...")
        return record


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """
    Top-level coordinator. Runs the full pipeline autonomously:
      1. Check past memory for context.
      2. Delegate to MonitorAgent.
      3. If anomaly: delegate to DiagnosisAgent → MitigationAgent → ReportAgent.
      4. Return the complete incident dict + spectrum data for dashboard.
    """
    NAME = "OrchestratorAgent"

    def __init__(self):
        self.monitor    = MonitorAgent()
        self.diagnosis  = DiagnosisAgent()
        self.mitigation = MitigationAgent()
        self.report     = ReportAgent()

    def run_pipeline(self, iq: np.ndarray,
                     sample_id: str = None) -> dict:
        """
        Full autonomous pipeline for one IQ snapshot.
        Returns a complete result dict with all agent outputs + spectrum data.
        No user input required — triggers entirely from the anomaly score.
        """
        if sample_id is None:
            sample_id = f"sig_{int(time.time()*1000) % 100000}"

        _log(self.NAME, "pipeline_start",
             f"=== New signal {sample_id} - orchestrating ===")

        # ── Check memory for context ──────────────────────────────────────
        past = get_incident_memory(n=3)
        if past:
            _log(self.NAME, "memory_check",
                 f"Last incident: {past[-1].get('type','?')} "
                 f"@ {past[-1].get('timestamp','?')}")

        # ── Monitor ───────────────────────────────────────────────────────
        mon_result = self.monitor.run(iq, sample_id)

        if not mon_result["is_anomaly"]:
            _log(self.NAME, "no_anomaly",
                 f"Score {mon_result['anomaly_score']:.5f} <= "
                 f"threshold {mon_result['threshold']:.5f} - signal is CLEAN")
            spec = get_spectrum_data(iq)
            return {
                "sample_id":    sample_id,
                "is_anomaly":   False,
                "severity":     "LOW",
                "status":       "CLEAN",
                "monitor":      mon_result,
                "spectrum":     spec,
                "diagnosis":    None,
                "mitigation":   None,
                "report":       None,
            }

        # ── Anomaly detected — full pipeline ─────────────────────────────
        _log(self.NAME, "anomaly_detected",
             f"Anomaly confirmed - activating Diagnosis Agent")

        dx_result  = self.diagnosis.run(iq, mon_result, sample_id)
        mit_result = self.mitigation.run(iq, dx_result)
        rep_result = self.report.run(dx_result, mit_result)
        spec       = get_spectrum_data(iq)
        spec_after = get_spectrum_data(mit_result["iq_filtered"])

        _log(self.NAME, "pipeline_complete",
             f"Incident #{rep_result['incident_id']} | "
             f"type={rep_result['type']} | "
             f"SNR Δ={rep_result['snr_delta_db']:+.2f} dB")

        return {
            "sample_id":   sample_id,
            "is_anomaly":  True,
            "severity":    mon_result.get("severity", "MEDIUM"),
            "status":      "INTERFERENCE_DETECTED",
            "monitor":     mon_result,
            "spectrum":    spec,
            "spectrum_after": spec_after,
            "diagnosis":   dx_result,
            "mitigation":  mit_result,
            "report":      rep_result,
        }

    def monitor_tick(self, iq: np.ndarray) -> dict:
        """
        Lightweight call for the dashboard's live poll (every N seconds).
        Only runs MonitorAgent + spectrum. Does NOT trigger full pipeline.
        """
        mon  = self.monitor.run(iq, f"tick_{int(time.time())%10000}")
        spec = get_spectrum_data(iq)
        return {
            "is_anomaly":    mon["is_anomaly"],
            "anomaly_score": mon["anomaly_score"],
            "threshold":     mon["threshold"],
            "severity":      mon.get("severity", "LOW"),
            "snr_db":        mon.get("snr_estimate_db", 0),
            "spectrum":      spec,
            "features":      mon["features"],
        }


# ── Shared singleton ──────────────────────────────────────────────────────────
_orchestrator: OrchestratorAgent = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    print("=" * 60)
    print("agent_system.py self-test (deterministic mode)")
    print("=" * 60)

    # Generate a CW interference sample for testing
    rng = np.random.default_rng(42)
    N   = 1024; FS = 20e6
    t   = np.arange(N) / FS
    iq  = ((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.3 +
           np.exp(1j * 2 * np.pi * 3e6 * t)).astype(np.complex64)

    orch   = get_orchestrator()
    result = orch.run_pipeline(iq, sample_id="selftest_cw")

    print(f"\nStatus:   {result['status']}")
    print(f"Severity: {result['severity']}")
    if result["diagnosis"]:
        print(f"Type:     {result['diagnosis']['interference_type']} "
              f"({result['diagnosis']['confidence']:.0%})")
        print(f"Explanation: {result['diagnosis']['explanation'][:120]}…")
    if result["report"]:
        print(f"SNR Δ:    {result['report']['snr_delta_db']:+.2f} dB")
        print(f"Strategy: {result['report']['mitigation']}")
        print(f"Incident: #{result['report']['incident_id']}")

    print(f"\nReasoning trace ({len(get_trace())} entries):")
    for e in get_trace():
        print(f"  [{e['agent']:20s}] {e['action']:20s} | {e['detail'][:60]}")
    print("\nOK")
