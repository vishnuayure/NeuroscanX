"""Data model for the XAI HUD – turns the structured summary into display rows (no Qt, fully testable).

Every value shown comes straight from the computed summary / local evidence."""
from __future__ import annotations

from ..xai.explanation import confidence_label

CONF_COLORS = {"HIGH": "#3ddc97", "MODERATE": "#ffc857", "LOW": "#ff6b81"}


def _lvl(v: float) -> str:
    return "HIGH" if v >= 0.7 else "MODERATE" if v >= 0.4 else "LOW"


def build_hud_model(result) -> dict:
    s = result.xai_summary
    ev, bd = result.evidence, result.boundary_analysis
    conf = s["segmentation"]["confidence"]
    rows = []
    mods = {k: v for k, v in result.modality_contribution.items() if not k.startswith("_")}
    for name, m in sorted(mods.items(), key=lambda kv: -kv[1]["contribution"]):
        rows.append({"name": name, "value": m["relative"], "level": m["level"], "detail": f"{m['contribution']:.0%}"})
    if ev.get("available"):
        rows.append({"name": "EDGE", "value": ev["edge_evidence"], "level": _lvl(ev["edge_evidence"]), "detail": f"{ev['edge_evidence']:.0%}"})
        rows.append({"name": "SPATIAL", "value": ev["spatial_evidence"], "level": _lvl(ev["spatial_evidence"]), "detail": f"{ev['spatial_evidence']:.0%}"})
    why = ""
    if ev.get("available"):
        top = [n for n, m in sorted(mods.items(), key=lambda kv: -kv[1]["contribution"]) if m["level"] != "LOW"]
        why = (f"{' + '.join(top[:3]) or 'Multimodal'} evidence; "
               f"{'spatially coherent region' if ev['largest_component_fraction'] >= 0.9 else 'fragmented region'}; "
               f"{'strong' if ev['edge_support'] >= 0.5 else 'moderate' if ev['edge_support'] >= 0.25 else 'weak'} boundary evidence.")
    bconf = bd.get("boundary_confidence") if bd.get("available") else None
    return {
        "title": "WHOLE TUMOR",
        "confidence": conf, "confidence_label": confidence_label(conf),
        "evidence": rows,
        "boundary_confidence": bconf, "boundary_label": confidence_label(bconf),
        "most_uncertain_side": bd.get("most_uncertain_side"),
        "why": why,
        "uncertainty_mean": s.get("uncertainty", {}).get("mean_uncertainty_tumor"),
        "uncertainty_label": s.get("uncertainty", {}).get("label", ""),
        "volume_ml": s.get("region", {}).get("volume_ml"),
        "method": s.get("method_label"), "explanation_source": result.explanation_source,
    }


def build_local_model(loc: dict) -> dict:
    if not loc or not loc.get("inside_brain"):
        return {"inside": False}
    rows = [{"name": n, "value": v, "z": loc["z"][n]} for n, v in sorted(loc["local_contribution"].items(), key=lambda kv: -kv[1])]
    return {"inside": True, "in_tumor": loc["in_tumor"], "membership": loc["tumor_membership"],
            "intensity_membership": loc["intensity_membership"], "uncertainty": loc["uncertainty"],
            "edge": loc["edge_strength"], "support": loc["neighbourhood_support"], "rows": rows,
            "distance_mm": loc["distance_to_boundary_mm"], "text": loc["local_explanation"]}
