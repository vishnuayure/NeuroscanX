"""Central, serialisable configuration for every NeuroScanX stage.

Every value that influences a result lives here so that runs are reproducible
and so that the exact configuration can be written into each case report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
METHODS = ("fcm", "spatial_fcm", "edge_spatial_fcm")
METHOD_LABELS = {
    "fcm": "FCM",
    "spatial_fcm": "Spatial FCM",
    "edge_spatial_fcm": "Edge-Aware SP-FCM",
}

DISCLAIMER = (
    "NeuroScanX is a research and decision-support prototype. Segmentation results "
    "are not a diagnosis and must not replace clinical judgment."
)


@dataclass
class PreprocessConfig:
    # --- brain / background handling -------------------------------------
    brain_mask_method: str = "auto"          # auto | nonzero | otsu
    brain_mask_closing: int = 2              # voxels, closes pin-holes in the mask
    crop_margin: int = 4                     # voxels of padding around the brain bounding box
    # --- denoising ---------------------------------------------------------
    denoise_method: str = "gaussian"         # gaussian | median | none
    denoise_sigma: float = 0.6               # voxels (gaussian) – deliberately small to keep edges
    denoise_median_size: int = 3
    # --- intensity normalisation ------------------------------------------
    norm_lower_percentile: float = 1.0
    norm_upper_percentile: float = 99.5
    # --- CLAHE -------------------------------------------------------------
    clahe_enabled: bool = True
    clahe_apply_to: str = "edges"            # edges | features | display | none  (see docs/README)
    clahe_clip_limit: float = 0.01
    clahe_kernel_fraction: float = 0.125     # kernel = fraction of slice size
    # --- edges -------------------------------------------------------------
    edge_method: str = "gradient"            # gradient | sobel
    edge_sigma_mm: float = 1.0               # gaussian derivative scale in mm (voxel-spacing aware)
    edge_combine: str = "max"                # max | mean | rms across modalities
    edge_modality_weights: dict = field(default_factory=lambda: {"T1": 1.0, "T1ce": 1.0, "T2": 1.0, "FLAIR": 1.0})
    edge_normalise_percentile: float = 99.0
    # --- feature construction ---------------------------------------------
    feature_modalities: tuple = MODALITIES


@dataclass
class FCMConfig:
    n_clusters: int = 6          # CSF, GM, WM, oedema, non-enhancing/necrotic, enhancing – tunable on BraTS
    m: float = 2.0               # fuzziness exponent (>1)
    tol: float = 1e-3            # max |ΔU| convergence threshold
    max_iter: int = 100
    seed: int = 42
    init_samples: int = 400_000  # voxels used for the (deterministic) global FCM fit / warm start
    # spatial penalty (Spatial FCM and Edge-Aware SP-FCM)
    spatial_weight: float = 0.6  # β – strength of the neighbourhood penalty relative to feature distance²
    connectivity: int = 6        # 6 | 18 | 26 neighbourhood
    relaxation: float = 0.7      # 1.0 = plain synchronous update, <1 damps voxel flip-flopping
    # edge-aware weights (Edge-Aware SP-FCM only)
    edge_penalty: float = 4.0    # λ – neighbour weight = exp(-λ · edge_ij)
    range_weight: float = 1.0    # exponent on the intensity-similarity (range) factor, 0 disables it
    range_scale: float = 3.0     # σ² of the range kernel = range_scale × mean squared neighbour difference
    # speed / memory
    downsample: int = 1          # cluster on a 1/downsample grid then refine at full resolution
    refine_iterations: int = 3   # full-resolution polishing iterations when downsample > 1
    eps: float = 1e-8


@dataclass
class TumorConfig:
    # cluster scoring weights (transparent, see segmentation/tumor_extraction.py)
    w_flair: float = 0.45
    w_t2: float = 0.30
    w_enh: float = 0.15
    w_t1_hypo: float = 0.10
    min_flair_z: float = 0.75     # a cluster must be at least this many robust-SDs above brain median on FLAIR …
    min_t2_z: float = 0.0         # … and not below the brain median on T2
    max_cluster_fraction: float = 0.35   # abnormal clusters larger than this fraction of the brain are diffuse, rejected
    relative_score: float = 0.35  # additional clusters must reach this fraction of the best score
    enclosed_fraction: float = 0.60    # a non-selected cluster joins if ≥ this fraction of its voxels lie inside the tumor (e.g. necrotic/enhancing core)
    min_enclosed_voxels: int = 30
    min_compactness: float = 0.30  # fraction of the cluster inside its largest connected component
    # post-processing
    membership_threshold: float = 0.5
    min_component_voxels: int = 50
    min_component_fraction: float = 0.10    # keep components ≥ this fraction of the largest one
    fill_holes: bool = True
    closing_radius: int = 1       # 0 disables morphological closing


@dataclass
class XAIConfig:
    uncertainty_measure: str = "linear"   # linear (index of fuzziness) | entropy
    boundary_band_voxels: int = 2
    uncertain_threshold: float = 0.5
    confident_threshold: float = 0.2
    high_level_rel: float = 0.75     # contribution / max contribution ≥ → HIGH
    moderate_level_rel: float = 0.40  # ≥ → MODERATE, else LOW
    ablation_max_voxels: int = 400_000
    use_llm: str = "auto"            # auto | off  (local Ollama only, never a cloud API)
    llm_model: Optional[str] = None


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    fcm: FCMConfig = field(default_factory=FCMConfig)
    tumor: TumorConfig = field(default_factory=TumorConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    device: str = "auto"             # auto | cpu | cuda
    reorient_to_ras: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
