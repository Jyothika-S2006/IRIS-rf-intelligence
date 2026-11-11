"""
mcp_server.py — IRIS MCP Tool Server
======================================
Exposes 4 IRIS agent tools as Model Context Protocol (MCP) tools so that
IBM Bob, LangChain, or any MCP-compatible client can call them directly.

Tools registered
----------------
  analyze_signal       — run autoencoder anomaly detection on IQ samples
  classify_signal      — classify interference type via GBM + CNN ensemble
  retrieve_kb          — semantic search over the RF interference knowledge base
  apply_mitigation     — apply DSP mitigation to IQ and return SNR improvement

Run standalone:
    python mcp_server.py

Or import and mount inside a FastAPI/Starlette app:
    from mcp_server import mcp_app
"""

import json
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── MCP server bootstrap ──────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    _mcp_available = True
except ImportError:
    _mcp_available = False
    print("[mcp_server] WARNING: 'mcp' package not installed. "
          "Install with: pip install mcp\n"
          "Falling back to plain function registry for testing.", flush=True)

if _mcp_available:
    mcp = FastMCP("IRIS-RF-Intelligence")
else:
    # Minimal stub so the rest of the file still imports cleanly
    class _Stub:
        def tool(self):
            def decorator(fn):
                return fn
            return decorator
        def run(self):
            print("[mcp_server] MCP not available — cannot start server.")
    mcp = _Stub()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — analyze_signal
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def analyze_signal(iq_real: list[float], iq_imag: list[float],
                   sample_id: str = "mcp_live") -> dict:
    """
    Run the IRIS autoencoder anomaly detector on an IQ signal.

    Parameters
    ----------
    iq_real : list of float  — real (I) channel samples
    iq_imag : list of float  — imaginary (Q) channel samples
    sample_id : str          — optional label for logging

    Returns a dict with anomaly_score, is_anomaly flag, severity, and features.
    """
    from agent_tools import analyze_signal as _analyze, validate_iq_input
    iq = (np.array(iq_real, dtype=np.float32) +
          1j * np.array(iq_imag, dtype=np.float32)).astype(np.complex64)
    valid, msg = validate_iq_input(iq)
    if not valid:
        return {"error": msg, "valid": False}
    result = _analyze(iq, sample_id=sample_id)
    # Make JSON-serialisable
    result.pop("features_raw", None)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — classify_signal
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def classify_signal(iq_real: list[float], iq_imag: list[float],
                    sample_id: str = "mcp_live") -> dict:
    """
    Classify the interference type in an IQ signal using the GBM + CNN ensemble.

    Returns interference_type, confidence, and model agreement flag.
    """
    from agent_tools import classify_signal as _classify, validate_iq_input
    iq = (np.array(iq_real, dtype=np.float32) +
          1j * np.array(iq_imag, dtype=np.float32)).astype(np.complex64)
    valid, msg = validate_iq_input(iq)
    if not valid:
        return {"error": msg, "valid": False}
    return _classify(iq, sample_id=sample_id)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — retrieve_kb
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def retrieve_kb(query: str, top_k: int = 3) -> dict:
    """
    Semantic search over the IRIS RF interference knowledge base.

    Returns the top-k text chunks most relevant to the query.
    Useful for grounding LLM explanations in documented RF engineering knowledge.
    """
    from agent_tools import retrieve_kb as _retrieve
    return _retrieve(query, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — apply_mitigation
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def apply_mitigation(iq_real: list[float], iq_imag: list[float],
                     interference_type: str) -> dict:
    """
    Apply the appropriate DSP mitigation to an IQ signal.

    interference_type must be one of:
        clean | narrowband_cw | wideband_noise | pulsed | adjacent_channel

    Returns SNR before/after, improvement_db, and effectiveness flag.
    """
    from agent_tools import apply_mitigation as _mitigate, validate_iq_input
    iq = (np.array(iq_real, dtype=np.float32) +
          1j * np.array(iq_imag, dtype=np.float32)).astype(np.complex64)
    valid, msg = validate_iq_input(iq)
    if not valid:
        return {"error": msg, "valid": False}
    result = _mitigate(iq, interference_type)
    # Remove raw IQ arrays from response (not JSON-friendly for MCP)
    result.pop("iq_filtered", None)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting IRIS MCP Tool Server…", flush=True)
    mcp.run()
