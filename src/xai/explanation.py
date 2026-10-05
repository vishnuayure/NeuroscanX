import numpy as np


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def explain_modality_contribution(
    contributions: dict[str, float]
) -> str:
    if not contributions:
        return "No modality contribution data is available."

    ranked = sorted(
        contributions.items(),
        key=lambda item: item[1],
        reverse=True
    )

    parts = [
        f"{modality} contributed {_format_percentage(score)}"
        for modality, score in ranked
    ]

    return "Modality contribution: " + ", ".join(parts) + "."


def explain_evidence(
    intensity_score: float,
    edge_score: float,
    spatial_score: float
) -> str:
    evidence = {
        "intensity": intensity_score,
        "edge": edge_score,
        "spatial": spatial_score
    }

    strongest = max(evidence, key=evidence.get)

    descriptions = {
        "intensity": "signal intensity",
        "edge": "boundary and edge information",
        "spatial": "spatial neighborhood information"
    }

    return (
        f"The strongest evidence came from "
        f"{descriptions[strongest]} "
        f"({_format_percentage(evidence[strongest])})."
    )


def explain_boundary(
    boundary_analysis: dict[str, float]
) -> str:
    if not boundary_analysis:
        return "No tumour boundary analysis is available."

    volume = boundary_analysis.get("tumor_volume", 0.0)
    density = boundary_analysis.get("boundary_density", 0.0)
    irregularity = boundary_analysis.get(
        "boundary_irregularity",
        0.0
    )

    if irregularity > 1.5:
        shape_description = "irregular"
    elif irregularity > 1.2:
        shape_description = "moderately irregular"
    else:
        shape_description = "relatively smooth"

    return (
        f"The detected tumour occupies approximately "
        f"{volume:.0f} voxels. Its boundary is {shape_description}, "
        f"with a boundary density of {density:.3f}."
    )


def generate_explanation(
    contributions: dict[str, float] | None = None,
    evidence: dict[str, float] | None = None,
    boundary_analysis: dict[str, float] | None = None
) -> str:
    sections = []

    if contributions:
        sections.append(
            explain_modality_contribution(contributions)
        )

    if evidence:
        sections.append(
            explain_evidence(
                evidence.get("intensity", 0.0),
                evidence.get("edge", 0.0),
                evidence.get("spatial", 0.0)
            )
        )

    if boundary_analysis:
        sections.append(
            explain_boundary(boundary_analysis)
        )

    if not sections:
        return "No explanation data is available."

    return " ".join(sections)


def explanation_report(
    contributions: dict[str, float] | None = None,
    evidence: dict[str, float] | None = None,
    boundary_analysis: dict[str, float] | None = None
) -> dict[str, object]:
    explanation = generate_explanation(
        contributions,
        evidence,
        boundary_analysis
    )

    return {
        "modality_contribution": contributions or {},
        "evidence": evidence or {},
        "boundary_analysis": boundary_analysis or {},
        "explanation": explanation
    }


__all__ = [
    "explain_modality_contribution",
    "explain_evidence",
    "explain_boundary",
    "generate_explanation",
    "explanation_report"
]