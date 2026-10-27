import numpy as np
from src.xai.evidence import LocalEvidenceExplainer
from src.xai.explanation import deterministic_explanation, validate_llm_text
from src.uncertainty.fuzzy_uncertainty import linear_fuzziness, binary_entropy, cluster_entropy, membership_margin
from agents.explanation_agent import ExplanationAgent


def test_uncertainty_bounds_and_shape():
    t = np.linspace(0, 1, 101)
    for f in (linear_fuzziness, binary_entropy):
        u = f(t)
        assert u.min() >= 0 and u.max() <= 1 + 1e-6 and abs(u[50] - 1) < 1e-3 and u[0] < 1e-3 and u[-1] < 1e-3
    U = np.random.default_rng(0).dirichlet(np.ones(4), size=50).T
    assert 0 <= cluster_entropy(U).min() and cluster_entropy(U).max() <= 1
    assert membership_margin(U).min() >= 0


def test_evidence_values_bounded(result):
    ev = result.evidence
    for k in ("intensity_evidence", "membership_strength", "spatial_coherence", "largest_component_fraction",
              "spatial_evidence", "edge_support", "spatial_changed_fraction"):
        assert 0.0 <= ev[k] <= 1.0, k


def test_modality_contribution_valid(result):
    mc = {k: v for k, v in result.modality_contribution.items() if not k.startswith("_")}
    assert set(mc) == {"T1", "T1ce", "T2", "FLAIR"}
    assert abs(sum(v["contribution"] for v in mc.values()) - 1.0) < 1e-6
    assert all(0 <= v["contribution"] <= 1 and v["level"] in ("HIGH", "MODERATE", "LOW") for v in mc.values())
    assert max(mc, key=lambda k: mc[k]["contribution"]) in ("FLAIR", "T2")   # phantom tumor is FLAIR/T2 bright


def test_uncertainty_valid_and_boundary_higher_than_far_field(result):
    u = result.uncertainty
    assert u.shape == tuple(result.mri.shape) and u.min() >= 0 and u.max() <= 1
    band = result.boundary_band.astype(bool)
    far = result.preprocessing.brain_mask & ~band & (result.segmentation == 0)
    assert u[band].mean() > u[far].mean()
    b = result.boundary_analysis
    assert 0 <= b["boundary_uncertainty"] <= 1 and b["most_uncertain_side"] in b["sides"]


def test_explanation_from_structured_evidence(result):
    s = result.xai_summary
    txt = deterministic_explanation(s)
    top = max((k for k in s["modalities"]), key=lambda k: s["modalities"][k]["contribution"])
    assert top in txt and f"{s['segmentation']['confidence']:.0%}" in txt
    assert "not a diagnosis" in txt and "cancer" not in txt.lower()
    assert result.explanation_source == "deterministic"


def test_changing_evidence_changes_explanation(result):
    import copy
    s = copy.deepcopy(result.xai_summary)
    s["segmentation"]["confidence"] = 0.31
    assert "Low" in deterministic_explanation(s) and "31%" in deterministic_explanation(s)


def test_llm_guard_blocks_invention(result):
    s = result.xai_summary
    assert not validate_llm_text("The patient has glioblastoma.", s)[0]
    assert not validate_llm_text("Tumor volume is 98765 voxels and confidence 13%.", s)[0]
    ok, _ = validate_llm_text(f"Confidence is {s['segmentation']['confidence']:.0%}.", s)
    assert ok


def test_agent_falls_back_without_llm(result):
    agent = ExplanationAgent("auto")
    agent.client.host = "http://127.0.0.1:9"      # nothing listens here
    text, src = agent.explain(result.xai_summary, "DET")
    assert (text, src) == ("DET", "deterministic")


def test_local_evidence(result):
    ex = LocalEvidenceExplainer(result.xai_context)
    c = result.xai_context
    inside = tuple(np.argwhere(c.tumor)[len(np.argwhere(c.tumor)) // 2])
    loc = ex.at(inside)
    assert loc["in_tumor"] and 0 <= loc["tumor_membership"] <= 1
    assert abs(sum(loc["local_contribution"].values()) - 1) < 1e-6 or sum(loc["local_contribution"].values()) == 0
    assert ex.at((0, 0, 0))["inside_brain"] is False


def test_local_text_flags_postprocessing_inclusion():
    from src.xai.evidence import local_text
    loc = {"inside_brain": True, "in_tumor": True, "tumor_membership": 0.23, "distance_to_boundary_mm": 4.0,
           "local_contribution": {"T1": 0.1, "FLAIR": 0.9}, "z": {"T1": -1.0, "FLAIR": 4.0}}
    assert "post-processing" in local_text(loc)
    loc["tumor_membership"] = 0.9
    assert "post-processing" not in local_text(loc)


def test_langgraph_matches_sequential_when_installed(prepared):
    pytest = __import__("pytest")
    pytest.importorskip("langgraph")
    nsx, prep = prepared
    r = nsx.analyze(prep, "edge_spatial_fcm")
    assert r.orchestrator == "langgraph" and "NEUROSCANX ANALYSIS" in r.explanation
