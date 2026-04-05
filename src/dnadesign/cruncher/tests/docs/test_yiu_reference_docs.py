"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_reference_docs.py

Reference-page contracts for payload-centric YIU docs surfaces.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_yiu_spec_reference_stays_schema_scoped() -> None:
    spec_ref = _read("docs/reference/yiu_spec.md")

    assert "This page owns schema and normalization only." in spec_ref
    assert "This page owns the input contract, normalization rules, and optimization rules." in spec_ref
    assert "metadata.source_workspace" in spec_ref
    assert "Ambiguous or missing sources fail fast." in spec_ref
    assert "`yiu/spec_models.py` is the stable public schema facade" in spec_ref
    assert "`yiu/payload_resolution.py` is the stable public input-resolution seam" in spec_ref


def test_yiu_artifacts_reference_stays_bundle_scoped() -> None:
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")

    assert "split_yiu_payload_bundle_v4" in artifacts_ref
    assert "visual_inventory.json" in artifacts_ref
    assert "shared `render`/`show` inspection surface" in artifacts_ref
    assert "render-status semantics" in artifacts_ref
    assert "bundle truth vs mirror" in artifacts_ref.lower()


def test_yiu_visual_system_describes_bench_strip_hierarchy() -> None:
    visual_ref = _read("docs/reference/yiu_visual_system.md")

    assert "`bench_strip`" in visual_ref
    assert "`payload` uses `evidence_ribbon`" in visual_ref
    assert "`split_payload` uses `operator_strip`" in visual_ref
    assert "`assembled_payload` uses `operator_strip`" in visual_ref
    assert "should share the same `bench_strip` foundation" in visual_ref


def test_yiu_architecture_names_boundary_seams() -> None:
    architecture = _read("docs/reference/architecture.md")

    assert "`yiu/` (payload-centric YIU domain)" in architecture
    assert "stable `yiu/spec_models.py` facade" in architecture
    assert "`yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, and `yiu/spec_rendering_models.py`" in architecture
    assert "`yiu/payload_resolution.py`" in architecture
    assert "`yiu/sample_hit_sources.py`" in architecture
    assert "named visual directions and style profiles live in `yiu/visual_system.py`" in architecture
