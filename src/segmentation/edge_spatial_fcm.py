"""
Alias module for Edge-Aware Spatially Penalized Fuzzy C-Means.
Re-exports EdgeSpatialFCM and helper functions from src.segmentation.sp_fcm.
"""
from src.segmentation.sp_fcm import (
    EdgeSpatialFCM,
    SPFCM,
    edge_spatial_fuzzy_c_means,
    sp_fuzzy_c_means,
    determine_tumour_cluster,
    generate_tumour_mask,
)

__all__ = [
    "EdgeSpatialFCM",
    "SPFCM",
    "edge_spatial_fuzzy_c_means",
    "sp_fuzzy_c_means",
    "determine_tumour_cluster",
    "generate_tumour_mask",
]