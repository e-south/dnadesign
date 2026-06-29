"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/models.py

Neutral data models for browser structure views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StructureFormat = Literal["pdb", "mmcif"]


@dataclass(frozen=True)
class StructureViewModel:
    """One structure model to load into a browser viewer."""

    model_id: str
    structure_text: str
    structure_format: StructureFormat = "pdb"
    label: str = ""
    color: str = "#0072B2"
    opacity: float = 1.0

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("StructureViewModel.model_id is required")
        if not self.structure_text.strip():
            raise ValueError(f"StructureViewModel.structure_text is required for {self.model_id}")
        if self.structure_format not in {"pdb", "mmcif"}:
            raise ValueError(f"Unsupported structure format for {self.model_id}: {self.structure_format}")
        if not (0.0 < float(self.opacity) <= 1.0):
            raise ValueError(f"StructureViewModel.opacity must be in (0, 1] for {self.model_id}")


@dataclass(frozen=True)
class StructureViewSelectionStyle:
    """Residue-level style overlay for one loaded structure model."""

    selection_id: str
    model_id: str
    label: str
    residue_numbers: tuple[int, ...]
    color: str = "#D55E00"
    opacity: float = 1.0

    def validate(self, *, model_ids: set[str]) -> None:
        if not self.selection_id.strip():
            raise ValueError("StructureViewSelectionStyle.selection_id is required")
        if self.model_id not in model_ids:
            raise ValueError(f"Selection style references unknown model_id: {self.model_id}")
        if not self.label.strip():
            raise ValueError(f"StructureViewSelectionStyle.label is required for {self.selection_id}")
        if not self.residue_numbers:
            raise ValueError(f"StructureViewSelectionStyle.residue_numbers is required for {self.selection_id}")
        if any(int(residue) < 1 for residue in self.residue_numbers):
            raise ValueError(f"Residue numbers must be one-based positive integers for {self.selection_id}")
        if not (0.0 < float(self.opacity) <= 1.0):
            raise ValueError(f"StructureViewSelectionStyle.opacity must be in (0, 1] for {self.selection_id}")


@dataclass(frozen=True)
class StructureViewSpec:
    """Backend-independent browser structure-view specification."""

    title: str
    models: tuple[StructureViewModel, ...]
    subtitle: str = ""
    selection_styles: tuple[StructureViewSelectionStyle, ...] = ()
    width: int = 700
    height: int = 500
    background_color: str = "#ffffff"
    style: str = "cartoon"

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("StructureViewSpec.title is required")
        if not self.models:
            raise ValueError("StructureViewSpec.models must contain at least one model")
        if int(self.width) < 240 or int(self.height) < 180:
            raise ValueError("StructureViewSpec width/height are too small for review use")
        if self.style not in {"cartoon", "line", "stick"}:
            raise ValueError(f"Unsupported structure-view style: {self.style}")
        for model in self.models:
            model.validate()
        model_ids = {model.model_id for model in self.models}
        for selection_style in self.selection_styles:
            selection_style.validate(model_ids=model_ids)
