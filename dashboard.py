"""
dashboard.py — IRIS — Interference Recognition & Intelligence System
===================================================
Premium red/black Streamlit dashboard.
Run:  streamlit run dashboard.py

Layout
------
  Header      — project name + live status indicator
  KPI row     — signals analysed, interference detected, avg confidence, active alerts
  Main (2-col)
    Left       — Waterfall/spectrogram + PSD spectrum plot (Plotly, dark/red)
    Right      — Classification card + Mitigation panel
  Alert timeline
  Agent Activity panel — live reasoning trace
  Simulate    — inject chosen interference type for live demo
  Chat        — ask the agent a free-form question
"""

import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IRIS — RF Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — red/black premium theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background-color: #0A0A0A !important;
    color: #F5F5F5 !important;
}

/* ── Header ── */
.rf-header {
    background: linear-gradient(135deg, #0A0A0A 0%, #1A0000 100%);
    border-bottom: 2px solid #E50914;
    padding: 18px 28px 14px 28px;
    margin-bottom: 20px;
    border-radius: 0 0 12px 12px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.rf-header h1 {
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #F5F5F5;
    margin: 0;
}
.rf-header .subtitle {
    font-size: 0.75rem;
    color: #9A9A9A;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Pulsing status dot ── */
.status-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}
.status-dot.live {
    background: #E50914;
    box-shadow: 0 0 0 0 rgba(229,9,20,0.7);
    animation: pulse-red 1.8s infinite;
}
.status-dot.idle {
    background: #9A9A9A;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0   rgba(229,9,20,0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(229,9,20,0); }
    100% { box-shadow: 0 0 0 0   rgba(229,9,20,0); }
}

/* ── KPI cards ── */
.kpi-card {
    background: #141414;
    border: 1px solid #2A2A2A;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    min-height: 100px;
}
.kpi-card:hover {
    border-color: #E50914;
    box-shadow: 0 0 14px rgba(229,9,20,0.18);
}
.kpi-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9A9A9A;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 900;
    color: #F5F5F5;
    line-height: 1;
}
.kpi-value.red  { color: #E50914; }
.kpi-value.amber{ color: #F59E0B; }
.kpi-value.green{ color: #22C55E; }

/* ── Panel cards ── */
.panel-card {
    background: #141414;
    border: 1px solid #2A2A2A;
    border-radius: 14px;
    padding: 18px 20px 16px 20px;
    margin-bottom: 16px;
}
.panel-card:hover {
    border-color: #8B0000;
    box-shadow: 0 0 12px rgba(139,0,0,0.15);
}
.panel-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #E50914;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2A2A2A;
}

/* ── Severity badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-high   { background: rgba(229,9,20,0.18);  color: #E50914; border: 1px solid #E50914; }
.badge-medium { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid #F59E0B; }
.badge-low    { background: rgba(154,154,154,0.12);color: #9A9A9A; border: 1px solid #3A3A3A; }
.badge-clean  { background: rgba(34,197,94,0.12);  color: #22C55E; border: 1px solid #22C55E; }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: #2A2A2A;
    border-radius: 4px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    background: linear-gradient(90deg, #8B0000, #E50914);
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s;
}

/* ── Alert timeline row ── */
.alert-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    background: #1A1A1A;
    border-left: 3px solid #2A2A2A;
    font-size: 0.82rem;
}
.alert-row.high   { border-left-color: #E50914; }
.alert-row.medium { border-left-color: #F59E0B; }
.alert-row.low    { border-left-color: #3A3A3A; }

/* ── Agent trace ── */
.trace-entry {
    display: flex;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'Courier New', monospace;
    margin-bottom: 4px;
    background: #111111;
    border-left: 2px solid #2A2A2A;
}
.trace-entry.orchestrator { border-left-color: #E50914; }
.trace-entry.monitoragent { border-left-color: #F59E0B; }
.trace-entry.diagnosisagent { border-left-color: #7C3AED; }
.trace-entry.mitigationagent { border-left-color: #0EA5E9; }
.trace-entry.reportagent { border-left-color: #22C55E; }
.trace-entry.toollayer { border-left-color: #9A9A9A; }
.trace-agent { font-weight: 700; min-width: 140px; color: #9A9A9A; }
.trace-action { color: #F5F5F5; }
.trace-detail { color: #9A9A9A; font-size: 0.68rem; }
.trace-ts { color: #555; font-size: 0.65rem; min-width: 80px; }

/* ── Simulate button ── */
.sim-button > button {
    background: #E50914 !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 0 18px rgba(229,9,20,0.45) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: box-shadow 0.2s !important;
}
.sim-button > button:hover {
    box-shadow: 0 0 30px rgba(229,9,20,0.7) !important;
}

/* ── Chat ── */
.chat-msg-user { text-align: right; margin: 6px 0; }
.chat-msg-user span {
    background: #8B0000; color: #F5F5F5;
    padding: 8px 14px; border-radius: 14px 14px 4px 14px;
    display: inline-block; max-width: 80%; font-size: 0.85rem;
}
.chat-msg-agent { text-align: left; margin: 6px 0; }
.chat-msg-agent span {
    background: #1A1A1A; color: #F5F5F5;
    padding: 8px 14px; border-radius: 14px 14px 14px 4px;
    display: inline-block; max-width: 80%; font-size: 0.85rem;
    border: 1px solid #2A2A2A;
}

/* ── Misc ── */
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 900 !important; }
.stSelectbox > div { background: #1A1A1A !important; border-color: #3A3A3A !important; }
.stTextInput input { background: #1A1A1A !important; border-color: #3A3A3A !important; }
hr { border-color: #2A2A2A !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "signals_analyzed":      0,
        "interference_detected": 0,
        "confidence_history":    [],
        "alerts":                [],
        "last_result":           None,
        "current_iq":            None,
        "is_monitoring":         False,
        "chat_history":          [],
        "auto_run":              False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ═══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS (avoid crashing if models not loaded yet)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _get_orchestrator():
    try:
        from agent_system import get_orchestrator
        return get_orchestrator()
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def _get_model_status():
    try:
        from agent_tools import models_available
        return models_available()
    except Exception:
        return {"gbm": False, "autoencoder": False, "cnn": False}

@st.cache_data(ttl=5, show_spinner=False)
def _get_trace():
    try:
        from agent_tools import get_trace
        return get_trace(last_n=60)
    except Exception:
        return []

@st.cache_data(ttl=30, show_spinner=False)
def _get_incidents(n=10):
    try:
        from agent_tools import get_incident_memory
        return get_incident_memory(n)
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATORS (for live demo & simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def _gen_signal(itype: str = "clean") -> np.ndarray:
    """Generate a fresh IQ sample of the requested type."""
    try:
        import importlib, simulator
        importlib.reload(simulator)   # pick up any changes
        gen_fn = {
            "clean":            simulator.gen_clean,
            "narrowband_cw":    simulator.gen_narrowband_cw,
            "wideband_noise":   simulator.gen_wideband_noise,
            "pulsed":           simulator.gen_pulsed,
            "adjacent_channel": simulator.gen_adjacent_channel,
        }.get(itype, simulator.gen_clean)
        iq, _ = gen_fn()
        return iq
    except Exception:
        # Minimal fallback if simulator not importable yet
        rng = np.random.default_rng()
        return (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS (dark red/black theme)
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0E0E0E",
    plot_bgcolor ="#111111",
    font=dict(family="Inter, sans-serif", color="#9A9A9A", size=11),
    margin=dict(l=10, r=10, t=32, b=10),
    xaxis=dict(gridcolor="#1E1E1E", zerolinecolor="#2A2A2A"),
    yaxis=dict(gridcolor="#1E1E1E", zerolinecolor="#2A2A2A"),
)

# Custom waterfall colorscale: black → dark red → bright red → white
WATERFALL_COLORSCALE = [
    [0.0,  "#000000"],
    [0.25, "#2D0000"],
    [0.5,  "#8B0000"],
    [0.75, "#E50914"],
    [1.0,  "#FFFFFF"],
]


def _psd_figure(freq: list, psd_db: list, title: str = "Power Spectral Density") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq, y=psd_db,
        mode="lines",
        line=dict(color="#E50914", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(139,0,0,0.15)",
        name="PSD (dBFS)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=12, color="#9A9A9A")),
        xaxis_title="Normalised Frequency",
        yaxis_title="dBFS",
        height=220,
    )
    return fig


def _waterfall_figure(spec: list, title: str = "Spectrogram (STFT Waterfall)") -> go.Figure:
    z = np.array(spec, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale=WATERFALL_COLORSCALE,
        showscale=False,
        xgap=0, ygap=0,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=12, color="#9A9A9A")),
        xaxis_title="Time →",
        yaxis_title="Freq →",
        height=220,
    )
    return fig


def _before_after_figure(spec_before: list, spec_after: list) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Before Mitigation", "After Mitigation"])
    for col, spec in enumerate([spec_before, spec_after], start=1):
        fig.add_trace(go.Heatmap(
            z=np.array(spec, dtype=float),
            colorscale=WATERFALL_COLORSCALE,
            showscale=False,
        ), row=1, col=col)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=200,
        title=dict(text="Spectrogram: Before vs After Mitigation",
                   font=dict(size=12, color="#9A9A9A")),
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.color = "#9A9A9A"
        ann.font.size  = 10
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _severity_badge(severity: str) -> str:
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium",
           "LOW": "badge-low", "CLEAN": "badge-clean"}.get(severity, "badge-low")
    return f'<span class="badge {cls}">{severity}</span>'


def _conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    color = "#E50914" if pct >= 75 else "#F59E0B" if pct >= 50 else "#9A9A9A"
    return (
        f'<div style="font-size:0.68rem;color:{color};margin-bottom:2px">{pct}%</div>'
        f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
        f'style="width:{pct}%;background:{color}"></div></div>'
    )


def _type_icon(itype: str) -> str:
    return {
        "clean":            "✅",
        "narrowband_cw":    "📍",
        "wideband_noise":   "🌊",
        "pulsed":           "⚡",
        "adjacent_channel": "↔️",
    }.get(itype, "❓")


def _render_trace(entries: list):
    if not entries:
        st.markdown(
            '<div class="trace-entry"><span class="trace-detail">No agent activity yet. '
            'Run a signal analysis to see the reasoning trace.</span></div>',
            unsafe_allow_html=True)
        return
    html_parts = []
    for e in reversed(entries[-30:]):
        agent_key = e["agent"].lower().replace(" ", "")
        ts_short  = e["ts"][11:19] if len(e["ts"]) >= 19 else e["ts"]
        html_parts.append(
            f'<div class="trace-entry {agent_key}">'
            f'<span class="trace-ts">{ts_short}</span>'
            f'<span class="trace-agent">{e["agent"]}</span>'
            f'<span class="trace-action">{e["action"]}</span>'
            f'<span class="trace-detail"> │ {e["detail"][:90]}</span>'
            f'</div>'
        )
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
model_status = _get_model_status()
any_model    = any(model_status.values())
dot_class    = "live" if any_model else "idle"
status_txt   = "MODELS LOADED — LIVE" if any_model else "MODELS NOT LOADED"

st.markdown(f"""
<div class="rf-header">
  <div>
    <h1>📡 IRIS — Interference Recognition & Intelligence System</h1>
    <p class="subtitle">Interference Recognition & Intelligence System · IBM watsonx.ai Granite · Multi-Agent AI</p>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
    <span class="status-dot {dot_class}"></span>
    <span style="font-size:0.7rem;letter-spacing:0.12em;color:#9A9A9A;text-transform:uppercase">
      {status_txt}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model status pills ─────────────────────────────────────────────────────────
cols_pills = st.columns(3)
for col, (name, ok) in zip(cols_pills, model_status.items()):
    label = name.upper().replace("_", " ")
    color = "#22C55E" if ok else "#E50914"
    icon  = "●" if ok else "○"
    col.markdown(
        f'<div style="text-align:center;font-size:0.65rem;font-weight:700;'
        f'letter-spacing:0.12em;color:{color}">{icon} {label}</div>',
        unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
incidents = _get_incidents(100)
total_sig  = st.session_state["signals_analyzed"]
total_int  = st.session_state["interference_detected"]
avg_conf   = (sum(st.session_state["confidence_history"]) /
              max(len(st.session_state["confidence_history"]), 1))
active_alr = sum(1 for a in st.session_state["alerts"]
                 if a.get("severity") == "HIGH")

k1, k2, k3, k4 = st.columns(4)
for col, label, value, val_class in [
    (k1, "Signals Analysed",      total_sig,         ""),
    (k2, "Interference Detected", total_int,         "red" if total_int > 0 else ""),
    (k3, "Avg Confidence",        f"{avg_conf:.0%}", ""),
    (k4, "Active Alerts",         active_alr,        "red" if active_alr > 0 else "green"),
]:
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {val_class}">{value}</div>'
        f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main two-column area ───────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # ── Spectrum plots ─────────────────────────────────────────────────────
    st.markdown('<div class="panel-card"><div class="panel-title">📊 Spectrum Analysis</div>',
                unsafe_allow_html=True)

    result = st.session_state.get("last_result")
    if result and result.get("spectrum"):
        spec_data = result["spectrum"]
        psd_fig   = _psd_figure(spec_data["freq_norm"], spec_data["psd_db"])
        st.plotly_chart(psd_fig, use_container_width=True, config={"displayModeBar": False})

        wf_fig = _waterfall_figure(spec_data["spectrogram"])
        st.plotly_chart(wf_fig, use_container_width=True, config={"displayModeBar": False})

        # Before/after if mitigation was applied
        if result.get("spectrum_after"):
            ba_fig = _before_after_figure(
                spec_data["spectrogram"], result["spectrum_after"]["spectrogram"])
            st.plotly_chart(ba_fig, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.markdown(
            '<div style="height:180px;display:flex;align-items:center;justify-content:center;'
            'color:#3A3A3A;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase">'
            'No signal loaded — use the simulate panel below</div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # ── Classification card ────────────────────────────────────────────────
    st.markdown('<div class="panel-card"><div class="panel-title">🏷️ Classification</div>',
                unsafe_allow_html=True)

    if result and result.get("diagnosis"):
        dx       = result["diagnosis"]
        itype    = dx["interference_type"]
        conf     = dx["confidence"]
        severity = result.get("severity", "MEDIUM")
        icon     = _type_icon(itype)

        st.markdown(
            f'<div style="font-size:1.4rem;font-weight:900;margin-bottom:6px">'
            f'{icon} {itype.replace("_"," ").title()}</div>'
            f'{_severity_badge(severity)}'
            f'{_conf_bar(conf)}'
            f'<div style="margin-top:10px;font-size:0.78rem;color:#9A9A9A;line-height:1.5">'
            f'{dx.get("explanation","")[:280]}</div>',
            unsafe_allow_html=True)

        # GBM vs CNN agreement
        gbm_r = dx.get("gbm_result", {})
        if gbm_r:
            agree_icon = "✅" if gbm_r.get("agreement") else "⚠️"
            st.markdown(
                f'<div style="margin-top:8px;font-size:0.68rem;color:#9A9A9A">'
                f'{agree_icon} GBM: <b style="color:#F5F5F5">{gbm_r.get("gbm_class","?")}</b> '
                f'({gbm_r.get("gbm_confidence",0):.0%}) &nbsp;|&nbsp; '
                f'CNN: <b style="color:#F5F5F5">{gbm_r.get("cnn_class","N/A")}</b> '
                f'({gbm_r.get("cnn_confidence",0):.0%}) &nbsp;|&nbsp; '
                f'Rechecks: {dx.get("rechecks",0)}</div>',
                unsafe_allow_html=True)
    elif result and result.get("status") == "CLEAN":
        st.markdown(
            f'<div style="font-size:1.2rem;font-weight:700">'
            f'✅ CLEAN SIGNAL</div>'
            f'<div style="font-size:0.75rem;color:#9A9A9A;margin-top:6px">'
            f'Anomaly score {result["monitor"]["anomaly_score"]:.5f} — '
            f'below threshold {result["monitor"]["threshold"]:.5f}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#3A3A3A;font-size:0.8rem;text-transform:uppercase;'
            'letter-spacing:0.1em">Awaiting signal…</div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Mitigation panel ───────────────────────────────────────────────────
    st.markdown('<div class="panel-card"><div class="panel-title">🔧 Mitigation</div>',
                unsafe_allow_html=True)

    if result and result.get("mitigation"):
        mit     = result["mitigation"]
        verify  = mit["verification"]
        eff_col = "#22C55E" if mit["effective"] else "#F59E0B"
        eff_txt = "EFFECTIVE" if mit["effective"] else "PARTIAL"

        st.markdown(
            f'<div style="font-size:0.85rem;font-weight:700;margin-bottom:8px">'
            f'{mit["strategy_applied"].replace("_"," ").upper()}</div>'
            f'<div style="font-size:0.72rem;color:#9A9A9A;margin-bottom:10px">'
            f'{mit.get("granite_rationale","")[:200]}</div>',
            unsafe_allow_html=True)

        # Before/after metrics
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="kpi-label">SNR Before</div>'
            f'<div style="font-size:1.1rem;font-weight:700">'
            f'{verify["before"]["snr_estimate_db"]:.1f} dB</div>',
            unsafe_allow_html=True)
        m2.markdown(
            f'<div class="kpi-label">SNR After</div>'
            f'<div style="font-size:1.1rem;font-weight:700">'
            f'{verify["after"]["snr_estimate_db"]:.1f} dB</div>',
            unsafe_allow_html=True)
        m3.markdown(
            f'<div class="kpi-label">Improvement</div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{eff_col}">'
            f'{verify["snr_improvement_db"]:+.1f} dB</div>',
            unsafe_allow_html=True)

        st.markdown(
            f'<div style="margin-top:10px">'
            f'<span class="badge" style="background:rgba(34,197,94,0.12);'
            f'color:{eff_col};border:1px solid {eff_col}">{eff_txt}</span>'
            f'</div>',
            unsafe_allow_html=True)

        # KB grounding
        kb_docs = result["diagnosis"].get("kb_mitigation", [])
        if kb_docs:
            with st.expander("📚 KB Sources", expanded=False):
                for doc in kb_docs[:2]:
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#9A9A9A;'
                        f'padding:6px;border-left:2px solid #E50914;margin-bottom:6px">'
                        f'<b style="color:#F5F5F5">[{doc["source"]}]</b> '
                        f'{doc["text"][:200]}…</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#3A3A3A;font-size:0.8rem;text-transform:uppercase;'
            'letter-spacing:0.1em">No mitigation applied yet</div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Alert Timeline ─────────────────────────────────────────────────────────────
st.markdown('<div class="panel-card"><div class="panel-title">🔴 Alert Timeline</div>',
            unsafe_allow_html=True)

alerts = st.session_state["alerts"]
if not alerts:
    st.markdown(
        '<div style="color:#3A3A3A;font-size:0.8rem;text-transform:uppercase;'
        'letter-spacing:0.1em;padding:10px 0">No alerts yet</div>',
        unsafe_allow_html=True)
else:
    for a in reversed(alerts[-10:]):
        sev = a.get("severity", "LOW").lower()
        ts  = a.get("ts", "")
        st.markdown(
            f'<div class="alert-row {sev}">'
            f'<span style="color:#555;font-size:0.68rem;min-width:80px">{ts[11:19]}</span>'
            f'{_severity_badge(a.get("severity","LOW"))}'
            f'<span style="font-weight:700">{_type_icon(a.get("type",""))} '
            f'{a.get("type","unknown").replace("_"," ").title()}</span>'
            f'<span style="color:#9A9A9A;font-size:0.78rem">'
            f'conf={a.get("confidence",0):.0%} · '
            f'SNR Δ={a.get("snr_delta",0):+.1f} dB · '
            f'{a.get("mitigation","—")}</span>'
            f'</div>',
            unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Two-col: Agent Activity + Simulate ────────────────────────────────────────
act_col, sim_col = st.columns([3, 2], gap="large")

with act_col:
    st.markdown('<div class="panel-card"><div class="panel-title">🤖 Agent Activity — Reasoning Trace</div>',
                unsafe_allow_html=True)
    trace_entries = _get_trace()
    _render_trace(trace_entries)
    st.markdown('</div>', unsafe_allow_html=True)

with sim_col:
    # ── Simulate Interference Panel ────────────────────────────────────────
    st.markdown('<div class="panel-card"><div class="panel-title">⚡ Simulate Interference</div>',
                unsafe_allow_html=True)

    sim_type = st.selectbox(
        "Interference Type",
        options=["clean", "narrowband_cw", "wideband_noise", "pulsed", "adjacent_channel"],
        format_func=lambda x: {
            "clean":            "✅ Clean OFDM",
            "narrowband_cw":    "📍 Narrowband CW Tone",
            "wideband_noise":   "🌊 Wideband Noise",
            "pulsed":           "⚡ Pulsed / Radar",
            "adjacent_channel": "↔️ Adjacent Channel",
        }[x],
        key="sim_type_select",
    )

    st.markdown(
        '<div class="sim-button">',
        unsafe_allow_html=True)
    run_btn = st.button("▶ RUN FULL AGENT PIPELINE", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        orch = _get_orchestrator()
        if orch is None:
            st.error("Agent system failed to load. Check that all Python files are present.")
        else:
            with st.spinner("🤖 Agents running…"):
                iq     = _gen_signal(sim_type)
                sid    = f"demo_{sim_type}_{int(time.time())%10000}"
                result = orch.run_pipeline(iq, sample_id=sid)

            # Update session state
            st.session_state["signals_analyzed"]     += 1
            st.session_state["current_iq"]            = iq
            st.session_state["last_result"]           = result
            if result["is_anomaly"]:
                st.session_state["interference_detected"] += 1
                conf = result["diagnosis"]["confidence"] if result["diagnosis"] else 0
                st.session_state["confidence_history"].append(conf)
                rep  = result.get("report", {})
                st.session_state["alerts"].append({
                    "ts":         datetime.utcnow().isoformat(),
                    "type":       result["diagnosis"]["interference_type"],
                    "severity":   result["severity"],
                    "confidence": conf,
                    "snr_delta":  rep.get("snr_delta_db", 0),
                    "mitigation": rep.get("mitigation", ""),
                })
            # Force cache refresh for trace
            _get_trace.clear()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Past Incidents ─────────────────────────────────────────────────────
    if incidents:
        st.markdown('<div class="panel-card"><div class="panel-title">📂 Past Incidents (Memory)</div>',
                    unsafe_allow_html=True)
        for inc in reversed(incidents[-4:]):
            eff_icon = "✅" if inc.get("effective") else "⚠️"
            st.markdown(
                f'<div style="font-size:0.72rem;padding:6px 0;border-bottom:1px solid #1A1A1A">'
                f'<b style="color:#E50914">#{inc.get("incident_id","?")}</b> &nbsp;'
                f'<b>{_type_icon(inc.get("type",""))} {inc.get("type","?").replace("_"," ")}</b>'
                f'<span style="color:#9A9A9A"> · {inc.get("confidence",0):.0%} · '
                f'{inc.get("mitigation","—")} {eff_icon}</span>'
                f'</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Chat Panel ─────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-card"><div class="panel-title">💬 Ask the Agent</div>',
            unsafe_allow_html=True)

chat_history = st.session_state["chat_history"]
for msg in chat_history[-8:]:
    role_cls = "chat-msg-user" if msg["role"] == "user" else "chat-msg-agent"
    st.markdown(
        f'<div class="{role_cls}"><span>{msg["content"]}</span></div>',
        unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    chat_cols = st.columns([5, 1])
    user_q = chat_cols[0].text_input(
        "", placeholder="Ask about the last signal, interference type, or mitigation…",
        label_visibility="collapsed")
    submit = chat_cols[1].form_submit_button("Send")

if submit and user_q.strip():
    chat_history.append({"role": "user", "content": user_q})

    # Build grounded answer from last result + KB
    answer = _chat_answer(user_q, st.session_state.get("last_result"))
    chat_history.append({"role": "agent", "content": answer})
    st.session_state["chat_history"] = chat_history
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT ANSWER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _chat_answer(question: str, result: dict) -> str:
    """
    Generate a grounded answer using Granite (if available) or rule-based fallback.
    Always sources claims from tool outputs / KB — never invents measurements.
    """
    # Build context from last result
    ctx_parts = []
    if result:
        ctx_parts.append(f"Last signal status: {result.get('status','unknown')}")
        if result.get("monitor"):
            ctx_parts.append(
                f"Anomaly score: {result['monitor']['anomaly_score']:.5f}, "
                f"threshold: {result['monitor']['threshold']:.5f}")
        if result.get("diagnosis"):
            dx = result["diagnosis"]
            ctx_parts.append(
                f"Interference type: {dx['interference_type']} "
                f"({dx['confidence']:.0%} confidence)")
            ctx_parts.append(f"Explanation: {dx.get('explanation','')[:200]}")
        if result.get("mitigation"):
            mit = result["mitigation"]
            ver = mit["verification"]
            ctx_parts.append(
                f"Mitigation: {mit['strategy_applied']}, "
                f"SNR improvement: {ver['snr_improvement_db']:+.2f} dB, "
                f"effective: {mit['effective']}")

    ctx = "\n".join(ctx_parts) if ctx_parts else "No signal has been analysed yet."

    # Try KB retrieval for grounding
    try:
        from agent_tools import retrieve_kb
        kb = retrieve_kb(question, top_k=2)
        kb_text = " | ".join(r["text"][:150] for r in kb["results"])
    except Exception:
        kb_text = ""

    # Try Granite
    try:
        from agent_system import _call_granite
        sys_p = (
            "You are the IRIS — Interference Recognition & Intelligence System. "
            "Answer ONLY using the provided signal data and KB excerpts. "
            "Be concise (2-3 sentences). Do not invent values."
        )
        usr_p = (
            f"Signal context:\n{ctx}\n\n"
            f"KB excerpts: {kb_text}\n\n"
            f"Question: {question}"
        )
        resp = _call_granite(sys_p, usr_p, max_tokens=256)
        if resp != "__DETERMINISTIC__":
            return resp
    except Exception:
        pass

    # Deterministic fallback
    q_lower = question.lower()
    if not result or result.get("status") == "CLEAN":
        return f"The last signal was CLEAN (no interference detected). {ctx}"
    if any(w in q_lower for w in ["what", "type", "kind", "interference"]):
        if result.get("diagnosis"):
            dx = result["diagnosis"]
            return (f"Detected: {dx['interference_type'].replace('_',' ')} "
                    f"with {dx['confidence']:.0%} confidence. {dx.get('explanation','')[:200]}")
    if any(w in q_lower for w in ["fix", "mitig", "solution", "how"]):
        if result.get("mitigation"):
            mit = result["mitigation"]
            ver = mit["verification"]
            return (f"Applied {mit['strategy_applied'].replace('_',' ')}. "
                    f"SNR improved by {ver['snr_improvement_db']:+.2f} dB. "
                    f"{'Effective.' if mit['effective'] else 'Partially effective.'} "
                    f"{mit.get('granite_rationale','')[:150]}")
    if any(w in q_lower for w in ["snr", "signal", "quality"]):
        if result.get("monitor"):
            return (f"SNR estimate: {result['monitor'].get('snr_estimate_db',0):.1f} dB. "
                    f"Anomaly score: {result['monitor']['anomaly_score']:.5f} "
                    f"(threshold {result['monitor']['threshold']:.5f}).")
    return f"Based on the last analysis: {ctx}"


st.markdown("---", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.65rem;color:#3A3A3A;'
    'letter-spacing:0.1em;padding:8px 0">'
    'IRIS — RF INTELLIGENCE · IBM watsonx.ai Granite · Simulated Data · 2024'
    '</div>',
    unsafe_allow_html=True)
