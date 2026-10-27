# NeuroScanX

**Multimodal MRI • Uncertainty-Aware Tumor Segmentation** — a native desktop research prototype.

> NeuroScanX is a research and decision-support prototype. Segmentation results are not a diagnosis and must not replace clinical judgment.

Whole-tumor segmentation from T1 / T1ce / T2 / FLAIR using **Edge-Aware Spatially Penalised Fuzzy C-Means** — no U-Net, nnU-Net, CNN, Transformer or SAM anywhere. BraTS labels are *only* used for evaluation.

```
MRI → validation → preprocessing → edge extraction → (FCM | Spatial FCM | Edge-Aware SP-FCM)
    → tumor-cluster identification → whole-tumor mask → uncertainty → XAI → explanation
    → 2D viewer · 3D mesh · HUD · dashboard · BraTS benchmark · export
```

## Run

```bash
pip install -r requirements.txt     # numpy scipy nibabel scikit-image PySide6 pyvista pyvistaqt
python main.py
```
Optional: `pip install langgraph` (LangGraph orchestration of the explanation layer), a CUDA build of `torch` (GPU), and a local [Ollama](https://ollama.com) server (LLM wording of the explanation). None is required.

Headless / library use:
```python
from src.pipeline import run
result = run(T1="t1.nii.gz", T1ce="t1ce.nii.gz", T2="t2.nii.gz", FLAIR="flair.nii.gz",
             method="edge_spatial_fcm",            # fcm | spatial_fcm | edge_spatial_fcm
             ground_truth="seg.nii.gz")            # optional → result.metrics
result.segmentation; result.membership; result.uncertainty; result.edges; result.preprocessing
result.modality_contribution; result.boundary_analysis; result.explanation; result.metrics
```
Tests: `python -m pytest` (59 tests; the 3D/GUI tests need a display — on a server use `QT_QPA_PLATFORM=xcb xvfb-run -a python -m pytest`).

## The algorithm

Features: the four modalities, robust-percentile-normalised inside the brain mask (denoised, cropped to the brain box). `edges`: multimodal 3-D Gaussian-derivative gradient magnitude, σ in **mm** (voxel-spacing aware), per-modality normalised, combined by max → soft boundary strength E ∈ [0,1].

| Method | Objective |
|---|---|
| **FCM** (Bezdek) | J = Σᵢₖ uₖᵢᵐ ‖xᵢ − vₖ‖² |
| **Spatial FCM** | J + (β / 2·n) Σᵢₖ uₖᵢᵐ Σ_{j∈Nᵢ} Σ_{l≠k} u_{lj}ᵐ  (Pham-style spatial penalty, wᵢⱼ = 1) |
| **Edge-Aware SP-FCM** | same penalty with wᵢⱼ = exp(−λ·½(Eᵢ+Eⱼ)) · exp(−ρ‖xᵢ−xⱼ‖² / 2σ²) |

The penalty is large when voxel *i* is pulled towards cluster *k* while its neighbours belong to other clusters. Closed-form update: `uₖᵢ ∝ (d²ₖᵢ + (β/n)·Pₖᵢ)^(−1/(m−1))`, `Pₖᵢ = Σⱼ wᵢⱼ (Sⱼ − uₖⱼᵐ)`; centres use the standard FCM update. Edge-aware weights shut the penalty off across boundaries, so memberships on either side of the tumor margin are not forced to agree. Initialisation is deterministic (seeded k-means++ on a voxel subsample); the spatial solvers are warm-started from that FCM fit and damped (`relaxation=0.7`) because synchronous updates otherwise flip-flop single voxels. All parameters live in `src/config.py` and are written into every report.

**Tumor-cluster identification** (`src/segmentation/tumor_extraction.py`): cluster ids are arbitrary, so each cluster is scored from its centre in robust-z units (`0.45·z_FLAIR + 0.30·z_T2 + 0.15·max(0,z_T1ce−z_T1) + 0.10·max(0,−z_T1)`), must pass eligibility gates (FLAIR hyper-intense, T2 not hypo-intense, not diffuse, spatially compact), and clusters enclosed by the tumor are re-included (necrotic / enhancing compartments). Every accept/reject reason is recorded and shown to the XAI layer. A unit test verifies that permuting cluster ids does not change the result. Post-processing: membership ≥ 0.5, drop tiny components, light closing, fill enclosed holes — voxels added only by post-processing are flagged in the "At crosshair" panel.

**Uncertainty**: tumor membership *t* = Σ memberships of tumor clusters; `uncertainty = 1 − |2t − 1|` (linear index of fuzziness, 0 = unambiguous, 1 = t = 0.5; binary entropy is available via `XAIConfig.uncertainty_measure`). Linear fuzziness is the default because FCM (m = 2) memberships are soft, and entropy saturates early (a clearly-tumor voxel at t = 0.86 would otherwise read as "58% uncertain"). Note that FCM memberships are **not calibrated probabilities**.

**XAI** (all numbers computed from the run): *modality contribution* = ½ distance-share (decomposition of the tumor-vs-normal squared feature distance) + ½ leave-one-modality-out ablation; *evidence* = intensity-only membership, neighbourhood/connected-component coherence, edge support on the margin, and how much spatial regularisation changed the decision; *boundary analysis* = band uncertainty, sharpness, and a direction-resolved (R/L/A/P/S/I) breakdown; *"why here?"* at the crosshair = per-voxel z-scores, local per-modality contribution, membership, uncertainty, distance to boundary.

**Explanation layer** (`agents/`): LangGraph (if installed; otherwise an identical sequential runner) runs evidence → uncertainty → modality contribution → boundary → summary → explanation. The text is generated deterministically from the structured summary. If a **local** Ollama model is reachable (loopback only, never a cloud API), it may rephrase it — but its output is rejected in favour of the deterministic text if it contains a number not present in the evidence or any diagnostic wording (tested). Only the JSON statistics, never image data, reach the LLM.

## Using the app
Drop four NIfTI files on the cards (optionally a BraTS segmentation → evaluation mode). Cards show dims, spacing, orientation, affine validity, detected modality and status; processing is disabled until the case validates. Volumes are reoriented to RAS+ internally and exports are written back **in the original voxel grid and affine of your input**.

2D viewer: wheel = slice · Ctrl+wheel = zoom · middle-drag = pan · right-drag = window/level · click/drag = crosshair (synchronised across axial/coronal/sagittal) · radiological convention (patient right on screen left). 3D view: left-drag rotate, wheel zoom, middle/shift-drag pan; SEGMENTATION / UNCERTAINTY (translucent shell coloured by local uncertainty around an opaque high-confidence core) / BOUNDARY (surface coloured by edge evidence). **Export case** writes `data/outputs/case_NNN/{segmentation,uncertainty,tumor_membership}.nii.gz, xai.json, report.json, processing.log, screenshots/` (+ comparison report if the benchmark was run).

## Measured behaviour — please read
* **Validated only on a synthetic phantom** (`tests/phantom.py`, test/developer use only) because no BraTS data was available when this was built. On it all three methods reach Dice ≈ 0.96–0.99, i.e. *the phantom cannot show that Edge-Aware SP-FCM beats the baselines.* What it does show: uniform Spatial FCM increases uncertainty on the tumor margin (it blurs it) while Edge-Aware SP-FCM keeps the FCM-level margin — the intended edge-preserving behaviour. **Proving the improvement requires running the BENCHMARK tab / `run_batch` on real BraTS cases**, and `n_clusters`, `spatial_weight`, `edge_penalty` should be tuned there (on training cases, evaluated on held-out ones).
* Performance (CPU, 240×240×155 phantom with 2.8 M brain voxels — about twice a typical skull-stripped brain): preprocessing 12 s · FCM 10 s · Edge-Aware SP-FCM 139 s full-resolution, 27 s in "Fast ½ grid + refinement" mode; peak RAM ≈ 2.3–2.5 GB.
* The **GPU (PyTorch/CUDA) path is implemented but untested** (no CUDA hardware here); any failure falls back to CPU automatically ("CUDA unavailable → falling back to CPU"). The **Ollama path is untested** (no model available); the deterministic path and the validation guard are tested. LangGraph and the sequential runner are both tested.
* Depth peeling is off by default in the 3D view (it blanked the scene on some renderers); set `NEUROSCANX_DEPTH_PEELING=1` to enable.
* Metrics are computed **inside the brain mask** (whole tumor = any BraTS label > 0); accuracy/specificity are therefore less inflated by background than whole-volume numbers.

## Layout
```
main.py            app/{ui,controllers,state}      src/{io,preprocessing,segmentation,uncertainty,xai,
agents/            tests/  scripts/  data/{raw,processed,outputs}      evaluation,visualization,pipeline,utils}
```
