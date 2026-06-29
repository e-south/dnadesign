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
class StructureViewSpec:
    """Backend-independent browser structure-view specification."""

    title: str
    models: tuple[StructureViewModel, ...]
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
