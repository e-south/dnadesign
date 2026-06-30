"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/runtime_fixtures.py

Shared runtime fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def mask_row(
    position: int,
    *,
    mapped: bool = True,
    motif: bool = False,
    protected: bool = False,
) -> dict[str, Any]:
    """Return a minimal mask row for structure-browser coordinate tests."""

    return {
        "canonical_position": position,
        "mapping_status": "mapped" if mapped else "unresolved_structure",
        "has_backbone_coordinates": mapped,
        "motif_protected": motif,
        "wang_ec86_direct_contact_prior": False,
        "direct_retained_dna_rna_contact_5a": False,
        "evolutionarily_conserved_clade9_25pct_plurality": False,
        "protected": motif or protected,
        "non_fixed": mapped and not motif and not protected,
    }


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    """Resolve a manifest-relative artifact path."""

    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


class FakeUi:
    """Minimal marimo UI facade for runtime render tests."""

    @staticmethod
    def table(rows: list[dict[str, str]], page_size: int) -> dict[str, Any]:
        return {"kind": "table", "rows": rows, "page_size": page_size}


class FakeMo:
    """Minimal marimo facade for runtime render tests."""

    ui = FakeUi()

    @staticmethod
    def md(value: str) -> str:
        return value

    @staticmethod
    def Html(value: str) -> str:
        return value

    @staticmethod
    def hstack(items: list[Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "hstack", "items": items, "kwargs": kwargs}

    @staticmethod
    def vstack(items: list[Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "vstack", "items": items, "kwargs": kwargs}

    @staticmethod
    def accordion(items: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"kind": "accordion", "items": items, "kwargs": kwargs}
