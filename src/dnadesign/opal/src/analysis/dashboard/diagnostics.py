"""Diagnostics helpers for dashboard workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Diagnostics:
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_note(self, message: str) -> "Diagnostics":
        return Diagnostics(
            notes=[*self.notes, message],
            warnings=list(self.warnings),
            errors=list(self.errors),
            meta=dict(self.meta),
        )

    def add_warning(self, message: str) -> "Diagnostics":
        return Diagnostics(
            notes=list(self.notes),
            warnings=[*self.warnings, message],
            errors=list(self.errors),
            meta=dict(self.meta),
        )

    def add_error(self, message: str) -> "Diagnostics":
        return Diagnostics(
            notes=list(self.notes),
            warnings=list(self.warnings),
            errors=[*self.errors, message],
            meta=dict(self.meta),
        )

    def merge(self, other: "Diagnostics") -> "Diagnostics":
        if other is None:
            return self
        return Diagnostics(
            notes=[*self.notes, *other.notes],
            warnings=[*self.warnings, *other.warnings],
            errors=[*self.errors, *other.errors],
            meta={**self.meta, **other.meta},
        )


def diagnostics_to_lines(diag: Diagnostics | None) -> list[str]:
    if diag is None:
        return []
    lines: list[str] = []
    if diag.notes:
        lines.extend(diag.notes)
    if diag.warnings:
        lines.extend([f"Warning: {msg}" for msg in diag.warnings])
    if diag.errors:
        lines.extend([f"Error: {msg}" for msg in diag.errors])
    return lines
