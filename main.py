import numpy as np

from src.io.nifti_loader import load_selected_brats_case
from src.preprocessing.normalization import normalize_volume
from src.preprocessing.denoising import denoise_volume
from src.preprocessing.clahe import clahe_enhance
from src.features.edge_detection import extract_edges
from src.segmentation.fcm import fuzzy_c_means
from src.segmentation.spatial_fcm import spatial_fuzzy_c_means
from src.segmentation.edge_spatial_fcm import edge_spatial_fuzzy_c_means
from src.uncertainty.uncertainty_map import calculate_uncertainty
from src.xai.evidence import analyze_evidence
from src.xai.modality_contribution import analyze_modality_contribution
from src.xai.boundary_analysis import analyze_boundary
from src.xai.explanation import generate_explanation
from src.visualization.viewer_2d import display_2d
from src.visualization.viewer_3d import display_3d
from src.visualization.xai_overlay import display_xai_overlay


def main():
    print("=" * 60)
    print("NeuroScanX")
    print("=" * 60)

    mri, ground_truth = load_selected_brats_case()

    print("\nMRI loaded successfully.")
    print(f"Case ID: {mri.case_id}")

    mri_data = {
        "T1": mri["t1"].get_data(dtype=np.float32),
        "T1ce": mri["t1ce"].get_data(dtype=np.float32),
        "T2": mri["t2"].get_data(dtype=np.float32),
        "FLAIR": mri["flair"].get_data(dtype=np.float32),
    }

    segmentation_ground_truth = ground_truth.get_data()

    print("\nMRI shapes:")

    for modality, volume in mri_data.items():
        print(f"{modality}: {volume.shape}")

    print("\nPreprocessing...")

    processed = {}

    for modality, volume in mri_data.items():
        normalized = normalize_volume(volume)
        denoised = denoise_volume(normalized)
        enhanced = clahe_enhance(denoised)
        processed[modality] = enhanced

    print("Preprocessing completed.")

    print("\nExtracting edge information...")

    edges = {}

    for modality, volume in processed.items():
        edges[modality] = extract_edges(volume)

    print("Edge extraction completed.")

    print("\nRunning FCM...")

    fcm_result = fuzzy_c_means(
        processed["FLAIR"]
    )

    print("FCM completed.")

    print("\nRunning Spatial FCM...")

    spatial_fcm_result = spatial_fuzzy_c_means(
        processed["FLAIR"]
    )

    print("Spatial FCM completed.")

    print("\nRunning Edge-Aware Spatial FCM...")

    edge_spatial_result = edge_spatial_fuzzy_c_means(
        processed["FLAIR"],
        edges["FLAIR"]
    )

    print("Edge-Aware Spatial FCM completed.")

    if isinstance(edge_spatial_result, tuple):
        segmentation = edge_spatial_result[0]

        if len(edge_spatial_result) > 1:
            membership = edge_spatial_result[1]
        else:
            membership = None
    else:
        segmentation = edge_spatial_result
        membership = None

    segmentation = np.asarray(segmentation)

    print("\nSegmentation shape:")
    print(segmentation.shape)

    print("\nCalculating uncertainty...")

    if membership is not None:
        uncertainty = calculate_uncertainty(
            membership
        )
    else:
        uncertainty = np.zeros_like(
            segmentation,
            dtype=np.float32
        )

    print("Uncertainty completed.")

    print("\nRunning XAI analysis...")

    try:
        evidence = analyze_evidence(
            processed,
            edges,
            segmentation
        )
    except TypeError:
        try:
            evidence = analyze_evidence(
                processed,
                segmentation
            )
        except TypeError:
            evidence = analyze_evidence(
                segmentation
            )

    try:
        modality_contribution = analyze_modality_contribution(
            processed,
            segmentation
        )
    except TypeError:
        modality_contribution = analyze_modality_contribution(
            processed
        )

    boundary_analysis = analyze_boundary(
        segmentation
    )

    try:
        explanation = generate_explanation(
            evidence=evidence,
            modality_contribution=modality_contribution,
            boundary_analysis=boundary_analysis
        )
    except TypeError:
        explanation = generate_explanation(
            evidence,
            modality_contribution,
            boundary_analysis
        )

    print("\nXAI completed.")

    print("\nBoundary Analysis:")

    for key, value in boundary_analysis.items():
        print(f"{key}: {value}")

    print("\nExplanation:")
    print(explanation)

    print("\nNeuroScanX pipeline completed.")

    return {
        "mri": mri,
        "ground_truth": segmentation_ground_truth,
        "processed": processed,
        "edges": edges,
        "fcm": fcm_result,
        "spatial_fcm": spatial_fcm_result,
        "segmentation": segmentation,
        "uncertainty": uncertainty,
        "evidence": evidence,
        "modality_contribution": modality_contribution,
        "boundary_analysis": boundary_analysis,
        "explanation": explanation,
    }


if __name__ == "__main__":
    main()