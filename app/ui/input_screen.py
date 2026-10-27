"""SCREEN 1 – case input: four modality cards (+ optional BraTS ground truth), validation, method selection."""
from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from app.controllers.workers import ValidationWorker
from src.config import METHODS, METHOD_LABELS, MODALITIES, PipelineConfig
from src.utils.backend import detect_device
from . import theme


class FileCard(QFrame):
    fileChanged = Signal(str, str)      # modality key, path

    def __init__(self, key: str, title: str, hint: str = "Select .nii / .nii.gz"):
        super().__init__()
        self.key, self.path = key, ""
        self.setObjectName("card")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(250, 190)
        lay = QVBoxLayout(self)
        self.title = QLabel(title); self.title.setObjectName("cardTitle")
        self.hint = QLabel(hint); self.hint.setObjectName("dim")
        self.info = QLabel(""); self.info.setObjectName("info"); self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.status = QLabel("—"); self.status.setObjectName("h")
        top = QHBoxLayout(); top.addWidget(self.title); top.addStretch(); top.addWidget(self.status)
        lay.addLayout(top); lay.addWidget(self.hint); lay.addWidget(self.info, 1)
        self._set_state("")

    def _set_state(self, st):
        self.setProperty("state", st)
        self.style().unpolish(self); self.style().polish(self)

    def set_path(self, path: str):
        self.path = path
        self.hint.setText(os.path.basename(path) if path else "Select .nii / .nii.gz")
        self.info.setText("validating…" if path else "")
        self.status.setText("…" if path else "—")
        self._set_state("")
        self.fileChanged.emit(self.key, path)

    def show_info(self, mi):
        """mi: ModalityInfo from the validation report."""
        if mi is None or mi.status == "missing":
            self.info.setText(""); self.status.setText("MISSING" if mi else "—"); self._set_state("error" if mi else "")
            return
        lines = [f"dims      {' × '.join(map(str, mi.shape)) or '—'}",
                 f"spacing   {' × '.join(f'{s:g}' for s in mi.spacing) or '—'} mm",
                 f"orient.   {mi.original_orientation or '?'} → {mi.orientation or '?'}",
                 f"affine    {'valid ✓' if mi.affine_valid else 'INVALID ✗'}"]
        if mi.name != "GT":
            lines.append(f"detected  {mi.detected_modality or 'unknown (by filename)'}")
        for i in mi.issues:
            if i.level != "info":
                lines.append(("✗ " if i.level == "error" else "⚠ ") + i.message)
        self.info.setText("\n".join(lines))
        col = {"ok": theme.OK, "warning": theme.WARN, "error": theme.DANGER}.get(mi.status, theme.DIM)
        self.status.setText(f"<span style='color:{col}'>{mi.status.upper()}</span>")
        self._set_state(mi.status)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            p, _ = QFileDialog.getOpenFileName(self, f"Select {self.key} NIfTI", "", "NIfTI (*.nii *.nii.gz);;All files (*)")
            if p:
                self.set_path(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self.setProperty("dragover", "true"); self.style().polish(self)

    def dragLeaveEvent(self, e):
        self.setProperty("dragover", "false"); self.style().polish(self)

    def dropEvent(self, e):
        self.setProperty("dragover", "false"); self.style().polish(self)
        urls = e.mimeData().urls()
        if urls:
            self.set_path(urls[0].toLocalFile())


class InputScreen(QWidget):
    runRequested = Signal(object, object, object, str, object)   # mri, report, gt_vol, method, config

    def __init__(self):
        super().__init__()
        self.mri = self.report = self.gt_vol = None
        self._worker: Optional[ValidationWorker] = None
        self.cards: Dict[str, FileCard] = {m: FileCard(m, m) for m in MODALITIES}
        self.gt_card = FileCard("GT", "GROUND TRUTH · OPTIONAL", "BraTS segmentation (evaluation mode)")
        lay = QVBoxLayout(self); lay.setContentsMargins(60, 36, 60, 10); lay.setSpacing(14)
        t = QLabel("NEUROSCANX"); t.setObjectName("title"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s = QLabel("MULTIMODAL MRI  •  UNCERTAINTY-AWARE TUMOR SEGMENTATION"); s.setObjectName("subtitle")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t); lay.addWidget(s); lay.addSpacing(18)
        grid = QGridLayout(); grid.setSpacing(14)
        for i, m in enumerate(MODALITIES):
            grid.addWidget(self.cards[m], 0, i)
            self.cards[m].fileChanged.connect(self._changed)
        lay.addLayout(grid)
        row = QHBoxLayout(); row.addWidget(self.gt_card, 1)
        self.gt_card.setMinimumHeight(120); self.gt_card.setMinimumWidth(250)
        self.gt_card.fileChanged.connect(self._changed)
        side = QVBoxLayout()
        self.summary = QLabel("Select T1, T1ce, T2 and FLAIR volumes (drag & drop or click)."); self.summary.setObjectName("info")
        self.summary.setWordWrap(True)
        side.addWidget(self.summary); side.addStretch()
        row.addLayout(side, 3)
        lay.addLayout(row)
        # settings
        st = QHBoxLayout(); st.setSpacing(12)
        st.addWidget(QLabel("METHOD")); self.method = QComboBox()
        for m in METHODS:
            self.method.addItem(METHOD_LABELS[m], m)
        self.method.setCurrentIndex(METHODS.index("edge_spatial_fcm")); st.addWidget(self.method)
        st.addWidget(QLabel("RESOLUTION")); self.res = QComboBox()
        self.res.addItem("Full resolution", 1); self.res.addItem("Fast · ½ grid + refinement", 2); st.addWidget(self.res)
        st.addWidget(QLabel("DEVICE")); self.device = QComboBox()
        self.device.addItem("Auto", "auto"); self.device.addItem("CPU", "cpu"); self.device.addItem("GPU (CUDA)", "cuda"); st.addWidget(self.device)
        self.dev_label = QLabel(detect_device("auto").message); self.dev_label.setObjectName("dim"); st.addWidget(self.dev_label)
        st.addStretch()
        self.run_btn = QPushButton("RUN NEUROSCANX"); self.run_btn.setObjectName("primary"); self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run); st.addWidget(self.run_btn)
        lay.addLayout(st)
        lay.addStretch()

    # ----------------------------------------------------------------------------------
    def paths(self) -> Dict[str, str]:
        return {m: c.path for m, c in self.cards.items() if c.path}

    def _changed(self, key, path):
        self.run_btn.setEnabled(False)
        self.mri = self.report = self.gt_vol = None
        if len(self.paths()) < len(MODALITIES):
            missing = [m for m in MODALITIES if m not in self.paths()]
            self.summary.setText("Waiting for: " + ", ".join(missing))
            return
        self.summary.setText("Validating case (readability, NaN/Inf, shape, spacing, affine, orientation, alignment)…")
        self._worker = ValidationWorker(self.paths(), self.gt_card.path or None, self)
        self._worker.done.connect(self._validated)
        self._worker.start()

    def _validated(self, mri, rep, gt, gt_info):
        if rep is None:
            self.summary.setText("Validation crashed:\n" + str(gt_info)); return
        self.mri, self.report, self.gt_vol = mri, rep, gt
        for m, c in self.cards.items():
            c.show_info(rep.modalities.get(m))
        if self.gt_card.path:
            self.gt_card.show_info(gt_info)
        msgs = [("✗ " + i.message) for i in rep.errors] + [("⚠ " + i.message) for i in rep.warnings]
        ok = rep.ok and (not self.gt_card.path or gt is not None)
        head = "<span style='color:%s'>CASE VALID</span>" % theme.OK if ok else "<span style='color:%s'>CASE REJECTED – processing disabled</span>" % theme.DANGER
        mode = " · evaluation mode (ground truth attached)" if (ok and gt is not None) else ""
        self.summary.setText(head + mode + ("<br>" + "<br>".join(dict.fromkeys(msgs)) if msgs else ""))
        self.run_btn.setEnabled(ok)

    def _run(self):
        cfg = PipelineConfig()
        cfg.fcm.downsample = int(self.res.currentData())
        cfg.device = self.device.currentData()
        self.dev_label.setText(detect_device(cfg.device).message)
        self.runRequested.emit(self.mri, self.report, self.gt_vol, self.method.currentData(), cfg)
