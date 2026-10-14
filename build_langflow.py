"""
build_langflow.py — Generate Langflow Flow JSON
================================================
This script generates a Langflow-compatible flow JSON that wires up the
multi-agent system with watsonx.ai Granite.

WHY WE GENERATE THE JSON programmatically:
  The Langflow UI is good for interactive design but not reproducible.
  By generating the JSON from Python, we can version-control the flow,
  recreate it on any Langflow instance, and parameterise it.

To import into Langflow:
  1. Run:  python build_langflow.py
  2. Open Langflow UI → Import → select langflow_flow.json
  3. Set your watsonx API key in the Langflow UI credentials panel.

The flow implements:
  Input (signal sample_id)
     ↓
  [Tool: analyze_signal]  → anomaly score
     ↓  (if anomaly)
  [Tool: classify_signal] → type + confidence
     ↓
  [Tool: retrieve_kb]     → grounding docs
     ↓
  [Granite LLM]           → explanation + mitigation recommendation
     ↓
  [Tool: apply_mitigation + re_run_detection] → before/after SNR
     ↓
  Output (structured incident report)
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).parent


def _node(type_name: str, display_name: str, x: int, y: int,
          data: dict = None) -> dict:
    """Create a Langflow node dict."""
    nid = str(uuid.uuid4())[:8]
    return {
        "id": f"{type_name.lower()}_{nid}",
        "type": "genericNode",
        "position": {"x": x, "y": y},
        "data": {
            "type":         type_name,
            "display_name": display_name,
            "node": {
                "template":    data or {},
                "description": display_name,
                "base_classes": ["Chain", "BaseLLM", "Tool"],
            },
        },
        "width": 240, "height": 160,
    }


def build_flow() -> dict:
    """Build the complete Langflow flow JSON."""
    nodes = []
    edges = []

    # ── Nodes ─────────────────────────────────────────────────────────────

    # 1. Granite LLM (watsonx.ai)
    granite = _node("WatsonxLLM", "Granite-13b-chat (watsonx.ai)", 800, 300, {
        "model_id":   {"value": "ibm/granite-13b-chat-v2", "type": "str"},
        "url":        {"value": "https://us-south.ml.cloud.ibm.com", "type": "str"},
        "project_id": {"value": "", "type": "str", "required": True},
        "api_key":    {"value": "", "type": "str", "required": True,
                       "password": True},
        "max_new_tokens":       {"value": 512,  "type": "int"},
        "temperature":          {"value": 0.2,  "type": "float"},
        "repetition_penalty":   {"value": 1.1,  "type": "float"},
    })

    # 2. Monitor Tool
    monitor_tool = _node("PythonFunctionTool", "Monitor: analyze_signal", 100, 100, {
        "name":        {"value": "analyze_signal", "type": "str"},
        "description": {"value": "Run autoencoder anomaly detection on IQ signal. "
                                 "Returns anomaly_score, threshold, is_anomaly, features.", "type": "str"},
        "code": {"value":
            "from agent_tools import analyze_signal as _fn\n"
            "import numpy as np, json\n"
            "def analyze_signal(sample_id: str) -> str:\n"
            "    from data_loader import load_iq_by_id\n"
            "    iq = load_iq_by_id(sample_id)\n"
            "    return json.dumps(_fn(iq, sample_id))\n",
            "type": "code"},
    })

    # 3. Classify Tool
    classify_tool = _node("PythonFunctionTool", "Diagnosis: classify_signal", 100, 300, {
        "name":        {"value": "classify_signal", "type": "str"},
        "description": {"value": "Classify interference type using GBM + CNN. "
                                  "Returns final_class, confidence, agreement, needs_recheck.", "type": "str"},
        "code": {"value":
            "from agent_tools import classify_signal as _fn\n"
            "import numpy as np, json\n"
            "def classify_signal(sample_id: str) -> str:\n"
            "    from data_loader import load_iq_by_id\n"
            "    iq = load_iq_by_id(sample_id)\n"
            "    return json.dumps(_fn(iq, sample_id))\n",
            "type": "code"},
    })

    # 4. KB Retrieval Tool
    kb_tool = _node("PythonFunctionTool", "RAG: retrieve_kb", 100, 500, {
        "name":        {"value": "retrieve_kb", "type": "str"},
        "description": {"value": "Semantic search over RF interference knowledge base. "
                                  "Input: query string. Returns grounding documents.", "type": "str"},
        "code": {"value":
            "from agent_tools import retrieve_kb as _fn\n"
            "import json\n"
            "def retrieve_kb(query: str) -> str:\n"
            "    return json.dumps(_fn(query, top_k=3))\n",
            "type": "code"},
    })

    # 5. Mitigation Tool
    mit_tool = _node("PythonFunctionTool", "Mitigation: apply + verify", 100, 700, {
        "name":        {"value": "apply_and_verify", "type": "str"},
        "description": {"value": "Apply DSP mitigation (notch/bandpass/blanking) to signal, "
                                  "re-run detection, return before/after SNR and anomaly score.", "type": "str"},
        "code": {"value":
            "from agent_tools import apply_mitigation, re_run_detection\n"
            "import numpy as np, json\n"
            "def apply_and_verify(sample_id: str, interference_type: str) -> str:\n"
            "    from data_loader import load_iq_by_id\n"
            "    iq = load_iq_by_id(sample_id)\n"
            "    mit = apply_mitigation(iq, interference_type)\n"
            "    ver = re_run_detection(iq, mit['filtered_iq'], sample_id)\n"
            "    return json.dumps({'strategy': mit['strategy'], "
            "'params': mit['params'], 'verification': ver})\n",
            "type": "code"},
    })

    # 6. Memory Tool — store incident
    store_tool = _node("PythonFunctionTool", "Memory: store_incident", 100, 900, {
        "name":        {"value": "store_incident", "type": "str"},
        "description": {"value": "Persist incident record to incidents.json memory store.", "type": "str"},
        "code": {"value":
            "from agent_tools import store_incident as _fn\n"
            "import json\n"
            "def store_incident(record_json: str) -> str:\n"
            "    record = json.loads(record_json)\n"
            "    return json.dumps(_fn(record))\n",
            "type": "code"},
    })

    # 7. Memory Tool — recall
    recall_tool = _node("PythonFunctionTool", "Memory: get_incidents", 100, 1100, {
        "name":        {"value": "get_incident_memory", "type": "str"},
        "description": {"value": "Retrieve last N incident records for context.", "type": "str"},
        "code": {"value":
            "from agent_tools import get_incident_memory as _fn\n"
            "import json\n"
            "def get_incident_memory(n: int = 5) -> str:\n"
            "    return json.dumps(_fn(n))\n",
            "type": "code"},
    })

    # 8. Agent Executor (orchestrator)
    agent_exec = _node("AgentExecutor", "RF Orchestrator Agent", 550, 600, {
        "system_message": {"value":
            "You are an autonomous RF interference detection and mitigation orchestrator. "
            "You have access to the following tools: analyze_signal, classify_signal, "
            "retrieve_kb, apply_and_verify, store_incident, get_incident_memory. "
            "\n\nYour workflow (ALWAYS follow this order):\n"
            "1. Call get_incident_memory to check for recent similar incidents.\n"
            "2. Call analyze_signal to get the anomaly score.\n"
            "3. If is_anomaly=True: call classify_signal.\n"
            "4. If needs_recheck=True: call classify_signal again.\n"
            "5. Call retrieve_kb with '<type> interference sources and characteristics'.\n"
            "6. Call retrieve_kb with '<type> mitigation strategy'.\n"
            "7. Call apply_and_verify with the interference type.\n"
            "8. Call store_incident with the full incident record.\n"
            "9. Return a structured report grounded ONLY in tool outputs.\n"
            "\nIMPORTANT: Never invent measurements. Every claim must reference a tool output.",
            "type": "str"},
        "max_iterations":       {"value": 12, "type": "int"},
        "handle_parsing_errors":{"value": True, "type": "bool"},
        "verbose":              {"value": True, "type": "bool"},
    })

    # 9. Text Input
    text_in = _node("TextInput", "Signal Sample ID", 50, 600, {
        "input_value": {"value": "narrowband_cw_0001", "type": "str"},
    })

    # 10. Text Output
    text_out = _node("TextOutput", "Incident Report", 1100, 600, {})

    nodes = [granite, monitor_tool, classify_tool, kb_tool, mit_tool,
             store_tool, recall_tool, agent_exec, text_in, text_out]

    # ── Edges ──────────────────────────────────────────────────────────────
    def edge(src_id, tgt_id, src_handle="output", tgt_handle="input"):
        return {
            "id":     f"e_{src_id[:8]}_{tgt_id[:8]}",
            "source": src_id, "sourceHandle": src_handle,
            "target": tgt_id, "targetHandle": tgt_handle,
            "animated": True,
        }

    edges = [
        edge(text_in["id"],      agent_exec["id"], "output", "input"),
        edge(granite["id"],      agent_exec["id"], "output", "llm"),
        edge(monitor_tool["id"], agent_exec["id"], "output", "tools"),
        edge(classify_tool["id"],agent_exec["id"], "output", "tools"),
        edge(kb_tool["id"],      agent_exec["id"], "output", "tools"),
        edge(mit_tool["id"],     agent_exec["id"], "output", "tools"),
        edge(store_tool["id"],   agent_exec["id"], "output", "tools"),
        edge(recall_tool["id"],  agent_exec["id"], "output", "tools"),
        edge(agent_exec["id"],   text_out["id"],   "output", "input"),
    ]

    return {
        "name":        "IRIS — Interference Recognition & Intelligence System",
        "description": "Autonomous RF interference detection, classification, "
                       "mitigation and reporting using watsonx.ai Granite",
        "data": {
            "nodes": nodes,
            "edges": edges,
            "viewport": {"x": 0, "y": 0, "zoom": 0.75},
        },
    }


if __name__ == "__main__":
    flow = build_flow()
    out  = ROOT / "langflow_flow.json"
    out.write_text(json.dumps(flow, indent=2))
    print(f"✓ Langflow flow JSON written → {out}")
    print(f"  Nodes: {len(flow['data']['nodes'])}")
    print(f"  Edges: {len(flow['data']['edges'])}")
    print()
    print("To import:")
    print("  1. Run Langflow:  langflow run")
    print("  2. UI → Import → langflow_flow.json")
    print("  3. Set WATSONX_APIKEY + WATSONX_PROJECT_ID in the Granite node")
