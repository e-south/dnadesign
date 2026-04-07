"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_reference_docs.py

Reference-page contracts for payload-centric YIU docs surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_yiu_spec_reference_stays_schema_scoped() -> None:
    spec_ref = _read("docs/reference/yiu_spec.md")

    assert "A YIU spec tells Cruncher where the payload comes from" in spec_ref
    assert (
        "The most common `sample_hit` handoff is a Sample public hit table such as "
        "`outputs/optimize/tables/elites.parquet`." in spec_ref
    )
    assert "metadata.source_workspace" in spec_ref
    assert "Ambiguous or missing sources fail fast." in spec_ref
    assert "### How the solver uses this spec" in spec_ref
    assert "does not change the biological search" in spec_ref
    assert "`yiu/spec_models.py` is the stable public schema facade" in spec_ref
    assert "`yiu/payload_resolution.py` is the stable public input-resolution seam" in spec_ref
    assert "`yiu/pwm_context.py` is the stable public PWM-resolution seam" in spec_ref
    assert "`yiu/pwm_context_sources.py`" in spec_ref
    assert "`yiu/pwm_context_sample_context.py`" in spec_ref
    assert "`yiu/pwm_context_sample_occurrences.py`" in spec_ref
    assert "`yiu/pwm_context_sample_motifs.py`" in spec_ref
    assert "## Ligation-aware mismatch ranking" in spec_ref
    assert "`ligation_profile=none` preserves legacy behavior." in spec_ref
    assert "derives mismatch class from the final duplex base pair" in spec_ref


def test_yiu_artifacts_reference_stays_bundle_scoped() -> None:
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")

    assert "split_yiu_payload_bundle_v4" in artifacts_ref
    assert "visual_inventory.json" in artifacts_ref
    assert "shared `render`/`show --verbose` inspection surface" in artifacts_ref
    assert "render-status semantics" in artifacts_ref
    assert "This page is only about the published bundle and the checks around it." in artifacts_ref
    assert "bundle truth vs mirror" in artifacts_ref.lower()
    assert "Treat that bundle directory as the source of truth." in artifacts_ref
    assert "`split_payload_view.jsonl` (JSONL)" in artifacts_ref
    assert "`PS` for the displayed payload strand and `AS` for the opposite strand" in artifacts_ref
    assert "reference duplex" in artifacts_ref
    assert "mismatch-present duplex" in artifacts_ref
    assert "one ligation summary line" in artifacts_ref
    assert "`views.payload`" in artifacts_ref
    assert "`views.split_left`" in artifacts_ref
    assert "`views.assembled`" in artifacts_ref
    assert "### Integrity checks" in artifacts_ref
    assert (
        "Human-readable `show --verbose` adds provenance, bundle contract, render and integrity detail,"
        in artifacts_ref
    )


def test_yiu_visual_system_describes_bench_strip_hierarchy() -> None:
    visual_ref = _read("docs/reference/yiu_visual_system.md")

    assert "`bench_strip`" in visual_ref
    assert "`payload` uses `evidence_ribbon`" in visual_ref
    assert "`split_payload` uses `operator_strip`" in visual_ref
    assert "`assembled_payload` uses `operator_strip`" in visual_ref
    assert "should share the same `bench_strip` foundation" in visual_ref
    assert "operator surface rather than a poster" in visual_ref
    assert "payload-forward coordinates `0..3`" in visual_ref
    assert "does not depend on whether the payload or complement strand was mutated" in visual_ref


def test_yiu_architecture_names_boundary_seams() -> None:
    architecture = _read("docs/reference/architecture.md")

    assert "`yiu/` (payload-centric YIU domain)" in architecture
    assert "stable `yiu/spec_models.py` facade" in architecture
    assert "`yiu/spec_input_models.py`, `yiu/spec_pwm_models.py`, and `yiu/spec_rendering_models.py`" in architecture
    assert "`yiu/payload_resolution.py`" in architecture
    assert "`yiu/sample_hit_sources.py`" in architecture
    assert "`yiu/pwm_context.py`" in architecture
    assert "`yiu/pwm_context_sources.py`" in architecture
    assert "`yiu/pwm_context_sample_context.py`" in architecture
    assert "`yiu/pwm_context_sample_occurrences.py`" in architecture
    assert "`yiu/pwm_context_sample_motifs.py`" in architecture
    assert "named visual-direction deltas live in `yiu/visual_directions.py`" in architecture
    assert "view registry and style profiles live in `yiu/visual_system.py`" in architecture
    assert "`yiu_payload_visual_v1.py` owns public adapter orchestration" in architecture
