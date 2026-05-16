"""Reference-set selector coverage for promoted promoter controls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.latentdna.src.references.sets import resolve_reference_set_ids_from_columns, resolve_reference_set_rows
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
    **extra: object,
) -> dict[str, object]:
    row = {
        "usr_label__primary": label,
        "source_family": source_family,
        "selection_basis": selection_basis,
        "promoter_standard__collection_id": collection_id,
        "promoter_standard__display_name": label.replace("_context1kb_forward", "")
        .replace("_context1kb_rc", "")
        .replace("_core60", ""),
    }
    row.update(extra)
    return row


def _matched(reference_set_id: str, rows: list[dict[str, object]]) -> list[str]:
    context = load_workspace_config(_stress_workspace())
    resolution = resolve_reference_set_rows(context.config.reference_sets[reference_set_id], rows)
    return resolution.matched_ids


def _matched_from_columns(reference_set_id: str, rows: list[dict[str, object]]) -> list[str]:
    context = load_workspace_config(_stress_workspace())
    columns = {
        column: [row.get(column) for row in rows]
        for column in {
            "usr_label__primary",
            "sfxi_ref__reference_instance_id",
            "derived__parent_id",
            "derived__parent_dataset",
            "regulondb__primary_promoter_name",
            "has_BaeR",
            "has_CpxR",
            "has_LexA",
            "promoter_standard__collection_id",
            "promoter_standard__display_name",
            "selection_basis",
            "source_family",
        }
    }
    resolution = resolve_reference_set_ids_from_columns(context.config.reference_sets[reference_set_id], columns)
    return resolution.matched_ids


def test_stress_reference_sets_resolve_native_core60_and_context_rows_without_mixing_sets() -> None:
    rows = [
        _row("spyp", selection_basis="legacy_construct_seed", source_family="legacy_construct_seed"),
        _row("sulAp", selection_basis="legacy_construct_seed", source_family="legacy_construct_seed"),
        _row("spyp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("sulAp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("spyp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("sulAp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("spyp_core60_context1kb_forward", selection_basis="template_window_center"),
        _row("sulAp_core60_context1kb_forward", selection_basis="template_window_center"),
        _row("spyp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("sulAp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("aceBp", selection_basis="native_source_length", source_family="legacy_reference_control"),
        _row("aceBp_context1kb_forward", selection_basis="template_window_center"),
        _row("aceBp_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("aceBp_core60", selection_basis="sigma_site_pair_midpoint"),
        _row("aceBp_core60_context1kb_forward", selection_basis="template_window_center"),
        _row("aceBp_core60_context1kb_rc", selection_basis="whole_output_reverse_complement"),
        _row("J23105", collection_id="anderson_igem", selection_basis="native_source_length"),
        _row("J23105_context1kb_forward", collection_id="anderson_igem", selection_basis="template_window_center"),
        _row("J23105_context1kb_rc", collection_id="anderson_igem", selection_basis="whole_output_reverse_complement"),
        _row("J23105_core60", collection_id="anderson_igem", selection_basis="sigma_site_pair_midpoint"),
        _row(
            "J23105_core60_context1kb_forward",
            collection_id="anderson_igem",
            selection_basis="template_window_center",
        ),
        _row(
            "J23105_core60_context1kb_rc",
            collection_id="anderson_igem",
            selection_basis="whole_output_reverse_complement",
        ),
        _row("W1", collection_id="t7_w_collection", selection_basis="native_source_length"),
        _row("W1_context1kb_forward", collection_id="t7_w_collection", selection_basis="template_window_center"),
        _row("W1_context1kb_rc", collection_id="t7_w_collection", selection_basis="whole_output_reverse_complement"),
        _row("W1_core60", collection_id="t7_w_collection", selection_basis="sigma_site_pair_midpoint"),
        _row(
            "W1_core60_context1kb_forward",
            collection_id="t7_w_collection",
            selection_basis="template_window_center",
        ),
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
        "spyp_core60_context1kb_forward",
        "sulAp_core60_context1kb_forward",
        "spyp_core60_context1kb_rc",
        "sulAp_core60_context1kb_rc",
    ]
    assert _matched("reference_native_mg1655", rows) == [
        "spyp",
        "sulAp",
        "spyp_context1kb_rc",
        "sulAp_context1kb_rc",
        "aceBp",
        "aceBp_context1kb_forward",
        "aceBp_context1kb_rc",
    ]
    assert _matched("reference_native_mg1655_core60", rows) == [
        "spyp_core60",
        "sulAp_core60",
        "spyp_core60_context1kb_forward",
        "sulAp_core60_context1kb_forward",
        "spyp_core60_context1kb_rc",
        "sulAp_core60_context1kb_rc",
        "aceBp_core60",
        "aceBp_core60_context1kb_forward",
        "aceBp_core60_context1kb_rc",
    ]
    assert _matched("reference_anderson_igem", rows) == [
        "J23105",
        "J23105_context1kb_forward",
        "J23105_context1kb_rc",
    ]
    assert _matched("reference_anderson_igem_core60", rows) == [
        "J23105_core60",
        "J23105_core60_context1kb_forward",
        "J23105_core60_context1kb_rc",
    ]
    assert _matched("reference_w_collection", rows) == ["W1", "W1_context1kb_forward", "W1_context1kb_rc"]
    assert _matched("reference_w_collection_core60", rows) == [
        "W1_core60",
        "W1_core60_context1kb_forward",
        "W1_core60_context1kb_rc",
    ]


def test_sfxi_archive_reference_set_is_backed_by_sfxi_overlay_metadata() -> None:
    rows = [
        _row("renamed_design", sfxi_ref__reference_instance_id="pDual-10-ES19p"),
        _row("pDual-10-ES99p", sfxi_ref__reference_instance_id=None),
    ]

    assert _matched("reference_sfxi_archive", rows) == ["pDual-10-ES19p"]


def test_regulondb_reference_sets_separate_full_native_cohort_from_tf_axis_subset() -> None:
    rows = [
        _row(
            "native_a_core60",
            source_family="regulondb_native_promoter",
            source_class="native_regulondb",
            derived__parent_id="usr_native_a",
            derived__parent_dataset="usr_regulondb_native_promoters",
            regulondb__primary_promoter_name="nativeA",
            has_BaeR=False,
            has_CpxR=False,
            has_LexA=False,
        ),
        _row(
            "native_b_core60",
            source_family="regulondb_native_promoter",
            source_class="native_regulondb",
            derived__parent_id="usr_native_b",
            derived__parent_dataset="usr_regulondb_native_promoters",
            regulondb__primary_promoter_name="nativeB",
            has_BaeR=False,
            has_CpxR=True,
            has_LexA=False,
        ),
        _row(
            "native_b_core60_context1kb_forward",
            source_family="construct_derived",
            source_class="manual_or_wildtype",
            derived__parent_id="usr_native_b",
            derived__parent_dataset="usr_regulondb_native_promoters",
            regulondb__primary_promoter_name="nativeB",
            has_BaeR=False,
            has_CpxR=True,
            has_LexA=False,
        ),
        _row(
            "native_c_core60_context1kb_forward",
            source_family="construct_derived",
            source_class="manual_or_wildtype",
            derived__parent_id="usr_native_c",
            derived__parent_dataset="usr_regulondb_native_promoters",
            regulondb__primary_promoter_name="nativeC",
            has_BaeR=False,
            has_CpxR=False,
            has_LexA=False,
        ),
        _row(
            "densegen_control",
            source_family="densegen_generated",
            source_class="densegen",
            derived__parent_id=None,
            derived__parent_dataset=None,
            regulondb__primary_promoter_name=None,
            has_BaeR=False,
            has_CpxR=False,
            has_LexA=False,
        ),
    ]

    assert _matched("reference_regulondb_native_core60_all", rows) == [
        "usr_native_a",
        "usr_native_b",
        "usr_native_c",
    ]
    assert _matched("reference_regulondb_tf_axis_targets", rows) == ["usr_native_b"]


def test_reference_set_column_resolution_matches_row_resolution_for_selector_sets() -> None:
    rows = [
        _row("W1", collection_id="t7_w_collection", selection_basis="native_source_length"),
        _row("W1_context1kb_forward", collection_id="t7_w_collection", selection_basis="template_window_center"),
        _row("W1_context1kb_rc", collection_id="t7_w_collection", selection_basis="whole_output_reverse_complement"),
        _row("W1_core60", collection_id="t7_w_collection", selection_basis="sigma_site_pair_midpoint"),
        _row(
            "W1_core60_context1kb_forward",
            collection_id="t7_w_collection",
            selection_basis="template_window_center",
        ),
        _row(
            "W1_core60_context1kb_rc",
            collection_id="t7_w_collection",
            selection_basis="whole_output_reverse_complement",
        ),
    ]

    assert _matched_from_columns("reference_w_collection", rows) == _matched("reference_w_collection", rows)
    assert _matched_from_columns("reference_w_collection_core60", rows) == _matched(
        "reference_w_collection_core60",
        rows,
    )


def test_reference_set_column_resolution_uses_positional_series_rows() -> None:
    context = load_workspace_config(_stress_workspace())
    reference_set = context.config.reference_sets["reference_w_collection"]
    columns = {
        "usr_label__primary": pd.Series(["W1", "W1_core60"], index=[10, 20]),
        "promoter_standard__collection_id": pd.Series(["t7_w_collection", "t7_w_collection"], index=[10, 20]),
        "promoter_standard__display_name": pd.Series(["W1", "W1 core60"], index=[10, 20]),
    }

    resolution = resolve_reference_set_ids_from_columns(reference_set, columns)

    assert resolution.matched_ids == ["W1"]


def test_reference_set_row_resolution_checks_required_columns_in_every_row() -> None:
    context = load_workspace_config(_stress_workspace())
    reference_set = context.config.reference_sets["reference_w_collection"]
    rows = [
        _row("W1", collection_id="t7_w_collection", selection_basis="native_source_length"),
        {"usr_label__primary": "W2"},
    ]

    resolution = resolve_reference_set_rows(reference_set, rows)

    assert resolution.complete is False
    assert "promoter_standard__collection_id" in resolution.missing_columns
