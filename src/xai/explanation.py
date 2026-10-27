"""Structured XAI summary + deterministic human-readable explanation.

The summary is the single source of truth: the deterministic text, the HUD and (optionally) a local LLM all
read from it.  Nothing here is invented – every sentence is generated from computed values.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from ..config import DISCLAIMER, METHOD_LABELS


def confidence_label(c: float | None) -> str:
    if c is None:
        return "Not available"
    return "High" if c >= 0.85 else "Moderate–High" if c >= 0.70 else "Moderate" if c >= 0.50 else "Low"


def uncertainty_label(u: float | None) -> str:
    if u is None:
        return "Not available"
    return "Low" if u < 0.15 else "Moderate" if u < 0.35 else "High"


def text_bar(value: float, width: int = 10) -> str:
    n = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * n + "░" * (width - n)


def build_summary(*, method: str, spacing, tumor_voxels: int, evidence: dict, uncertainty: dict, modalities: dict,
                  boundary: dict, extraction, fcm_info: dict) -> dict:
    vox_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    vol = tumor_voxels * vox_mm3
    conf = uncertainty.get("confidence")
    mods = {k: v for k, v in modalities.items() if not k.startswith("_")}
    return {
        "disclaimer": DISCLAIMER,
        "method": method, "method_label": METHOD_LABELS.get(method, method),
        "region": {"voxels": int(tumor_voxels), "volume_mm3": vol, "volume_ml": vol / 1000.0,
                   "tumor_clusters": list(extraction.selected), "postprocessing": extraction.postprocess},
        "segmentation": {"confidence": conf, "confidence_label": confidence_label(conf)},
        "uncertainty": {**uncertainty, "label": uncertainty_label(uncertainty.get("mean_uncertainty_tumor"))},
        "modalities": mods,
        "evidence": evidence,
        "boundary": boundary,
        "cluster_selection": extraction.cluster_report,
        "clustering": fcm_info,
        "warnings": list(extraction.warnings),
    }


def deterministic_explanation(s: dict) -> str:
    """Template-based explanation – used directly, and as the grounding/fallback for the optional LLM."""
    ev, bd, mods = s.get("evidence", {}), s.get("boundary", {}), s.get("modalities", {})
    if not ev.get("available"):
        return ("NEUROSCANX ANALYSIS\n\nNo region met the segmentation criteria for this case, so no explanation "
                "can be generated.\n\n" + s["disclaimer"])
    lines: List[str] = ["NEUROSCANX ANALYSIS", "", "Region consistent with the segmentation criteria, identified primarily from:"]
    ranked = sorted(mods.items(), key=lambda kv: -kv[1]["contribution"])
    for name, m in ranked:
        if m["level"] == "LOW":
            continue
        if name == "T1ce" and m.get("enhancement_z", 0) > 1.0:
            lines.append(f"• localized T1ce enhancement (+{m['enhancement_z']:.1f} z over T1; contribution {m['contribution']:.0%})")
        elif name == "T1" and m["direction"] == "reduced":
            lines.append(f"• reduced T1 signal (z={m['tumor_z']:+.1f}; contribution {m['contribution']:.0%})")
        else:
            lines.append(f"• {m['direction']} {name} signal (z={m['tumor_z']:+.1f}; contribution {m['contribution']:.0%})")
    if ev["largest_component_fraction"] >= 0.9 and ev["spatial_coherence"] >= 0.8:
        lines.append(f"• spatially coherent region ({ev['largest_component_fraction']:.0%} of the volume in one connected component)")
    else:
        lines.append(f"• region split over {ev['n_components']} components (largest holds {ev['largest_component_fraction']:.0%})")
    es = ev["edge_support"]
    lines.append(f"• {'strong' if es >= 0.5 else 'moderate' if es >= 0.25 else 'weak'} boundary evidence "
                 f"({es:.0%} of margin voxels in the top-quartile edge strength)")
    conf = s["segmentation"]["confidence"]
    lines += ["", "Confidence:", f"{confidence_label(conf)} ({conf:.0%} mean voxel confidence inside the region)"]
    u = s["uncertainty"]
    lines += ["", "Uncertainty:"]
    if bd.get("available"):
        side = bd.get("most_uncertain_side")
        lines.append(f"Boundary uncertainty {bd['boundary_uncertainty']:.0%}; "
                     f"{bd['fraction_uncertain_boundary']:.0%} of boundary voxels are in the uncertain class."
                     + (f" Highest around the {side} boundary ({bd['sides'][side]['mean_uncertainty']:.0%})." if side else ""))
    lines.append(f"{u['fraction_high_confidence']:.0%} of region voxels are high-confidence.")
    lines += ["", "Evidence:"]
    for name, m in ranked:
        lines.append(f"{name:<5} {text_bar(m['relative'])}  {m['level'].capitalize()}")
    lines.append(f"Edge  {text_bar(ev['edge_evidence'])}  {ev['edge_evidence']:.0%} support")
    lines.append(f"Spatial {text_bar(ev['spatial_evidence'])}  {ev['spatial_evidence']:.0%} coherence")
    if s.get("warnings"):
        lines += ["", "Notes:"] + [f"• {w}" for w in s["warnings"]]
    lines += ["", s["disclaimer"]]
    return "\n".join(lines)


# ---- guard used for LLM output ---------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")
_FORBIDDEN = re.compile(r"\b(cancer|glioblastoma|gbm|malignan\w*|benign|metastas\w*|carcinoma|lymphoma|you have|patient has|"
                        r"diagnos(?:is|ed) (?:of|with)|prognosis|survival|treatment|biopsy|stage \w+|grade \w+)\b", re.I)


def _numeric_leaves(o) -> Iterable[float]:
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from _numeric_leaves(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _numeric_leaves(v)


def validate_llm_text(text: str, summary: dict) -> tuple[bool, str]:
    """Reject LLM output that adds diagnoses or numbers that are not supported by the structured evidence."""
    if _FORBIDDEN.search(text):
        return False, "contains diagnostic / clinical wording"
    allowed = set()
    for v in _numeric_leaves({k: val for k, val in summary.items() if k != "disclaimer"}):
        allowed.update({v, abs(v), v * 100.0, abs(v) * 100.0, v * 1000.0, v / 1000.0})
    allowed.update({float(i) for i in range(0, 11)})   # small counts / scale words (e.g. "top 2", "4 modalities")
    for tok in _NUM.findall(text):
        x = abs(float(tok.replace("−", "-")))
        if not any(abs(x - a) <= max(0.51, 0.011 * abs(a)) for a in allowed):
            return False, f"unsupported number {tok}"
    return True, "ok"
