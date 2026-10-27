"""Everything the XAI modules need, in brain-box coordinates (all arrays share the same box)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..config import XAIConfig


@dataclass
class XAIContext:
    names: Tuple[str, ...]
    features: np.ndarray            # (D, *box)  normalised features used by the clustering
    centers: np.ndarray             # (C, D) final cluster centres
    membership: np.ndarray          # (C, *box) final fuzzy memberships
    selected: List[int]             # tumor clusters
    tumor_membership: np.ndarray    # (*box)
    tumor: np.ndarray               # (*box) bool – final whole-tumor mask
    brain: np.ndarray               # (*box) bool
    edges: np.ndarray               # (*box) edge strength [0,1]
    uncertainty: np.ndarray         # (*box) [0,1]
    feature_median: np.ndarray
    feature_scale: np.ndarray
    spacing: Tuple[float, float, float]
    m: float
    cluster_report: List[dict]
    cfg: XAIConfig = field(default_factory=XAIConfig)
    cache: Dict[str, object] = field(default_factory=dict)

    @property
    def D(self):
        return len(self.names)
