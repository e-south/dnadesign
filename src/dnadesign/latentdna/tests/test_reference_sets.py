"""Reference-set selector coverage for promoted promoter controls."""

from __future__ import annotations

from pathlib import Path

from dnadesign.latentdna.src.reference_sets import resolve_reference_set_rows
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _stress_workspace() -> Path:
    return _repo_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth"


def _row(
    label: str,
    *,
    collection_id: str | None = None,
    selection_basis: str = "template_window_center",
    source_family: str = "construct_derived",
) -> dict[str, object]:
    return {
        "usr_label__primary": label,
        "source_family": source_family,
        "selection_basis": selection_basis,
        "promoter_standard__collection_id": collection_id,
        "promoter_standard__display_name": label.replace("_context1kb_rc", "").replace("_core60", ""),
    }


def _matched(reference_set_id: str, rows: list[dict[str, object]]) -> list[str]:
    context = load_workspace_config(_stress_workspace())
    resolution = resolve_reference_set_rows(context.config.reference_sets[reference_set_id], rows)
    return resolution.matched_ids


def test_stress_reference_sets_resolve_native_core60_and_context_rows_without_mixing_sets() -> None:
    rows = [
        _row("spyp", selection_basis="legacy_construct_seed", source_family="legacy_construct_seed"),
        _row("sulAp", selection_basis="legacy_construct_seed", source_family="legacy_construct_seed"),
        _row("spyp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("sulAp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("spyp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("sulAp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("spyp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("sulAp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("aceBp", selection_basis="native_source_length", source_family="legacy_reference_control"),
        _row("aceBp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("aceBp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("aceBp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("J23105", collection_id="anderson_igem", selection_basis="native_source_length"),
        _row("J23105_context1kb_rc", collection_id="anderson_igem", selection_basis="whole_output_reverse_complement"),
        _row("J23105_core60", collection_id="anderson_igem", selection_basis="sigma_site_pair_midpoint"),
        _row(
            "J23105_core60_context1kb_rc",
            collection_id="anderson_igem",
            selection_basis="whole_output_reverse_complement",
        ),
        _row("W1", collection_id="t7_w_collection", selection_basis="native_source_length"),
        _row("W1_context1kb_rc", collection_id="t7_w_collection", selection_basis="whole_output_reverse_complement"),
        _row("W1_core60", collection_id="t7_w_collection", selection_basis="sigma_site_pair_midpoint"),
        _row(
            "W1_core60_context1kb_rc",
            collection_id="t7_w_collection",
            selection_basis="whole_output_reverse_complement",
        ),
    ]

    assert _matched("reference_spyp_sulap", rows) == [
        "spyp",
        "sulAp",
        "spyp_context1kb_rc",
        "sulAp_context1kb_rc",
    ]
    assert _matched("reference_spyp_sulap_core60", rows) == [
        "spyp_core60",
        "sulAp_core60",
        "spyp_core60_context1kb_rc",
        "sulAp_core60_context1kb_rc",
    ]
    assert _matched("reference_native_mg1655", rows) == ["aceBp", "aceBp_context1kb_rc"]
    assert _matched("reference_native_mg1655_core60", rows) == [
        "aceBp_core60",
        "aceBp_core60_context1kb_rc",
    ]
    assert _matched("reference_anderson_igem", rows) == ["J23105", "J23105_context1kb_rc"]
    assert _matched("reference_anderson_igem_core60", rows) == [
        "J23105_core60",
        "J23105_core60_context1kb_rc",
    ]
    assert _matched("reference_w_collection", rows) == ["W1", "W1_context1kb_rc"]
    assert _matched("reference_w_collection_core60", rows) == [
        "W1_core60",
        "W1_core60_context1kb_rc",
    ]
