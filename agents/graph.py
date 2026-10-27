"""Orchestration of the explanation layer.

Nodes (each a deterministic, local function from src/xai):
  evidence_extraction → uncertainty_analysis → modality_contribution → boundary_analysis → build_summary
  → explanation_generation (deterministic text, optionally rewritten by a validated local LLM)

LangGraph is used when installed; otherwise the identical node functions run in a plain sequential loop.
The medical-image computation itself never depends on LangGraph or on an LLM.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, TypedDict

from src.uncertainty.fuzzy_uncertainty import summarize_uncertainty
from src.xai.boundary_analysis import analyze_boundary
from src.xai.evidence import compute_global_evidence
from src.xai.explanation import build_summary, deterministic_explanation
from src.xai.modality_contribution import compute_modality_contribution
from .explanation_agent import ExplanationAgent

log = logging.getLogger("neuroscanx")


class XAIState(TypedDict, total=False):
    ctx: Any
    extraction: Any
    method: str
    fcm_info: dict
    tumor_voxels: int
    evidence: dict
    uncertainty: dict
    modalities: dict
    boundary: dict
    boundary_arrays: dict
    summary: dict
    explanation: str
    explanation_source: str
    node_times: dict


def _timed(name: str, fn: Callable[[XAIState], dict]) -> Callable[[XAIState], dict]:
    def wrapper(state: XAIState) -> dict:
        t = time.time()
        upd = fn(state)
        times = dict(state.get("node_times", {}))
        times[name] = time.time() - t
        upd["node_times"] = times
        return upd
    return wrapper


def n_evidence(s):      return {"evidence": compute_global_evidence(s["ctx"])}
def n_uncertainty(s):
    c = s["ctx"]
    return {"uncertainty": summarize_uncertainty(c.uncertainty, c.tumor, c.cfg.confident_threshold, c.cfg.uncertain_threshold)}
def n_modalities(s):    return {"modalities": compute_modality_contribution(s["ctx"])}
def n_boundary(s):
    b, arrays = analyze_boundary(s["ctx"])
    return {"boundary": b, "boundary_arrays": arrays}
def n_summary(s):
    c = s["ctx"]
    return {"summary": build_summary(method=s["method"], spacing=c.spacing, tumor_voxels=s["tumor_voxels"],
                                     evidence=s["evidence"], uncertainty=s["uncertainty"], modalities=s["modalities"],
                                     boundary=s["boundary"], extraction=s["extraction"], fcm_info=s["fcm_info"])}
def n_explain(s):
    det = deterministic_explanation(s["summary"])
    agent = ExplanationAgent(s["ctx"].cfg.use_llm, s["ctx"].cfg.llm_model)
    text, src = agent.explain(s["summary"], det)
    return {"explanation": text, "explanation_source": src}


NODES = [("evidence_extraction", n_evidence), ("uncertainty_analysis", n_uncertainty),
         ("modality_contribution", n_modalities), ("boundary_analysis", n_boundary),
         ("build_summary", n_summary), ("explanation_generation", n_explain)]


def _build_langgraph():
    from langgraph.graph import StateGraph, START, END   # optional dependency
    g = StateGraph(XAIState)
    prev = START
    for name, fn in NODES:
        g.add_node(name, _timed(name, fn))
        g.add_edge(prev, name)
        prev = name
    g.add_edge(prev, END)
    return g.compile()


def run_xai_graph(initial: XAIState) -> tuple[XAIState, str]:
    """Run the XAI graph; returns (final state, orchestrator name)."""
    try:
        app = _build_langgraph()
        final = app.invoke(initial)
        return final, "langgraph"
    except ImportError:
        pass
    except Exception as e:   # a LangGraph problem must never break the deterministic pipeline
        log.warning("LangGraph failed (%s) → sequential orchestration", e)
    state: XAIState = dict(initial)
    for name, fn in NODES:
        state.update(_timed(name, fn)(state))
    return state, "sequential"
