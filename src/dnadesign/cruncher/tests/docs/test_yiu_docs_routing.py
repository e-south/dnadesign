"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_docs_routing.py

Routing contracts for payload-centric YIU docs surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_top_level_docs_route_readers_to_yiu_surfaces() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "payload-centric YIU" in package_readme
    assert "`sample` and fixed-length optimization" in package_readme
    assert "docs/README.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme
    assert "docs/guides/sampling_and_analysis.md" in package_readme
    assert "sample_hit" in package_readme
    assert "This README stays light on purpose." in package_readme
    assert "[Docs map](docs/README.md): the comprehensive index." in package_readme

    for content in (docs_readme, docs_index):
        assert "Payload-Centric YIU Workflows" in content
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content
        assert "reference/yiu_spec.md" in content
        assert "reference/yiu_artifacts.md" in content
        assert "reference/yiu_visual_system.md" in content
        assert "YIU docs route" in content
        assert "yiu init-workspace|validate|render|show" in content
        assert "workspaces/demo_monotypic_tetr/runbook.md" in content
        assert "trace|solve" not in content


def test_cli_reference_lists_public_yiu_surface() -> None:
    cli_ref = _read("docs/reference/cli.md")

    assert "YIU workflows" in cli_ref
    assert "cruncher yiu init-workspace" in cli_ref
    assert "cruncher yiu validate" in cli_ref
    assert "cruncher yiu render" in cli_ref
    assert "cruncher yiu show" in cli_ref
    assert "split_yiu_payload_rendering_v4" in cli_ref
    assert "Treat the bundle directory as the source of truth" in cli_ref
    assert "`motif_context`, `optimization_decision`, and `split_row_debug`" in cli_ref
    assert "operator-facing handoff summary `bundle_summary.json`" in cli_ref
    assert "`split_payload_view.jsonl` (JSONL rows)" in cli_ref
    assert "one ligation summary line" in cli_ref


def test_yiu_workflow_routes_to_contract_pages() -> None:
    guide = _read("docs/guides/yiu_workflow.md")

    assert "YIU turns one payload sequence into a checked junction-mismatch bundle." in guide
    assert "### Where `sample_hit` comes from" in guide
    assert "[YIU Spec Reference](../reference/yiu_spec.md)" in guide
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in guide
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in guide
    assert "[Cruncher architecture](../reference/architecture.md)" in guide
    assert "[Sampling and Analysis](../guides/sampling_and_analysis.md)" in guide
    assert "Ambiguous or missing sources fail fast." in guide
    assert "Cross-tool integrations should not import `dnadesign.baserender.src.*`." in guide
    assert "Human-readable `--verbose` adds provenance, bundle contract, render/integrity details," in guide
    assert "The remaining published JSON files are machine-facing bundle ledgers or render contracts" in guide
    assert "reference duplex and mismatch-present duplex" in guide
    assert "### How YIU chooses a plan" in guide
    assert "Enumerate mismatch plans exhaustively" in guide
    assert (
        "Use `candidate_positions: [0, 1, 2, 3]` when you want ligation-aware ranking "
        "to compare edge and middle offsets." in guide
    )
    assert "### Ligation-aware mismatch ranking" in guide
    assert "Bilotti et al." in guide
    assert "`ligation_profile=t4` is the recommended default for T4-like assembly workflows." in guide
    assert "`ligation_selection_mode=hard_ligation_filter`" in guide
    assert "fails fast and prints a short relaxation hint" in guide
    assert "### How candidate counts work" in guide
    assert "YIU runs in three stages: generate candidates, apply the ligation policy, then rank the survivors." in guide
    assert "`before` is the full candidate pool" in guide
    assert "In `hard_ligation_filter`, PWM does not change either count." in guide
    assert "`count: 1` gives `4 positions × 2 strands × 3 bases = 24` candidates per window." in guide
    assert "`count: 2` gives `C(4,2) × 2^2 × 3^2 = 6 × 4 × 9 = 216` candidates per window." in guide
    assert '"feasible windows × per-window combinatorics"' in guide
    assert "### What strict mode removes" in guide
    assert "That leaves at most `5` survivors per window, or about `2.3%`" in guide
    assert "### Why PWM still changes the winner" in guide
    assert "PWM may prefer the `edge,middle` survivor" in guide
