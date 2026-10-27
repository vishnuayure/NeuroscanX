"""Shared UI state: current case, crosshair, display settings.  Widgets listen to the signals."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from src.config import MODALITIES
from src.visualization.viewer_2d import default_window


class AppState(QObject):
    resultChanged = Signal()
    crosshairChanged = Signal()
    displayChanged = Signal()

    def __init__(self):
        super().__init__()
        self.result = None
        self.crosshair: Tuple[int, int, int] = (0, 0, 0)
        self.modality = "FLAIR"
        self.wl: Dict[str, Tuple[float, float]] = {}
        self.seg_on = True
        self.unc_on = False
        self.edge_on = False
        self.boundary_on = False
        self.crosshair_on = True
        self.opacity = 0.45

    def set_result(self, result):
        self.result = result
        self.wl = {}
        for m, vol in result.mri.modalities.items():
            self.wl[m] = default_window(np.nan_to_num(vol.data))
        if self.modality not in result.mri.modalities:
            self.modality = next(iter(result.mri.modalities))
        seg = result.segmentation
        if seg.any():
            from scipy import ndimage as ndi
            c = ndi.center_of_mass(seg)
            self.crosshair = tuple(int(round(v)) for v in c)
        else:
            self.crosshair = tuple(s // 2 for s in seg.shape)
        self.resultChanged.emit()
        self.crosshairChanged.emit()

    def set_crosshair(self, ijk):
        shp = self.result.segmentation.shape
        self.crosshair = tuple(int(np.clip(v, 0, s - 1)) for v, s in zip(ijk, shp))
        self.crosshairChanged.emit()

    def set_modality(self, m):
        self.modality = m
        self.displayChanged.emit()

    def set_wl(self, center, width):
        self.wl[self.modality] = (float(center), max(float(width), 1e-6))
        self.displayChanged.emit()

    def update_display(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.displayChanged.emit()
