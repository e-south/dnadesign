"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/batch0/test_opal_batch0.py

Focused contracts for the stress/ethanol/ciprofloxacin OPAL batch-0 handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.opal import load_config
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table import (
    main as candidate_table_main,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table import (
    materialize_configured_candidate_feature_table,
    validate_candidate_feature_table,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.provenance import (
    audit_candidate_lineage,
    show_candidate_lineage,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.select import (
    REQUIRED_REVIEW_COLUMNS,
    load_sampling_config,
    select_batch0,
    validate_configured_candidate_feature_table,
    validate_selected_ids_against_candidate_feature_table,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.select import (
    main as batch0_select_main,
)


def repo_root_from(path: str | Path) -> Path:
    current = Path(path).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"repo root not found from {current}")


REPO_ROOT = repo_root_from(__file__)
SAMPLING = REPO_ROOT / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml"
CAMPAIGN_ROOT = REPO_ROOT / "src/dnadesign/opal/campaigns"
STUDY_DOCS = REPO_ROOT / "docs/studies/stress_ethanol_cipro_growth"
BATCH0_README = REPO_ROOT / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/README.md"


def _write_fixed_x_records(path: Path, data: dict[str, list[object]], *, x_col: str) -> None:
    values = data[x_col]
    first_vector = next((value for value in values if value is not None), None)
    if first_vector is None:
        raise ValueError("test fixture requires at least one non-null X vector")
    x_dim = len(first_vector)  # type: ignore[arg-type]
    columns: dict[str, pa.Array] = {
        column: pa.array(column_values) for column, column_values in data.items() if column != x_col
    }
    columns[x_col] = pa.array(values, type=pa.list_(pa.float32(), list_size=int(x_dim)))
    pq.write_table(pa.table(columns), path)


def _candidate_provenance(
    row_count: int,
    *,
    source_classes: list[str] | None = None,
    design_families: list[str] | None = None,
) -> dict[str, list[object]]:
    return {
        "opal_candidate__role": ["opal_candidate_feature_table"] * row_count,
        "opal_candidate__x_source_view_id": ["bidir"] * row_count,
        "opal_candidate__source_class": source_classes or ["densegen"] * row_count,
        "opal_candidate__design_family": design_families or ["ethanol"] * row_count,
        "opal_candidate__sfxi_ref__collection_id": [None] * row_count,
        "densegen__plan": [f"plan-{idx}" for idx in range(row_count)],
        "densegen__run_id": ["study_stress_ethanol_cipro"] * row_count,
        "densegen__sampling_library_hash": [f"hash-{idx}" for idx in range(row_count)],
        "densegen__used_tfbs_detail": [
            _densegen_tfbs_detail(("background", "baeR", "background")) for _ in range(row_count)
        ],
        "densegen__required_regulators": [["baeR"] for _ in range(row_count)],
    }


def _row(
    row_id: str,
    *,
    plan: str,
    regulators: str,
    tfbs_regulators: str | None = None,
    slot_pattern: tuple[str, str, str] | None = None,
    sigma: str = "f",
    spacer: int = 16,
    ethanol: float = 0.2,
    cipro: float = 0.2,
    dual: float = 0.2,
    tier: int = 1,
    sequence: str = "ACGT" * 15,
) -> dict[str, object]:
    if tfbs_regulators == "none":
        tfbs_summary = "none"
    elif slot_pattern is not None:
        tfbs_summary = ";".join(
            f"{regulator}@{idx * 20}" for idx, regulator in enumerate(slot_pattern) if regulator != "background"
        )
    else:
        tfbs_summary = ";".join(f"{regulator}@{spacer}" for regulator in (tfbs_regulators or regulators).split("+"))
    row: dict[str, object] = {
        "id": row_id,
        "sequence": sequence,
        "canonical_densegen_plan": plan,
        "regulator_composition": regulators,
        "sigma35_variant": sigma,
        "spacer_length": spacer,
        "target_margin": max(ethanol, cipro, dual),
        "synthetic_margin_ethanol_vs_background": ethanol,
        "synthetic_margin_cipro_vs_background": cipro,
        "synthetic_margin_dual_vs_background": dual,
        "sig35_margin_f_vs_b": 0.4,
        "tfbs_summary": tfbs_summary,
        "motif_score_summary": f"tier={tier}",
        "tfbs_offset_summary": str(spacer),
        "tfbs_orientation_summary": "fwd",
        "motif_tier_summary": str(tier),
        "x_provenance": "intermediate_embedding_7b_context_anchor_mean_bidir_concat",
    }
    if slot_pattern is not None:
        row["densegen__used_tfbs_detail"] = _densegen_tfbs_detail(slot_pattern)
    return row


def _densegen_tfbs_detail(pattern: tuple[str, str, str]) -> list[dict[str, object]]:
    return [
        {
            "part_kind": "tfbs",
            "regulator": regulator,
            "sequence": "ACGTACGTAC",
            "offset_raw": idx * 20,
            "offset": idx * 20,
            "length": 10,
            "end": idx * 20 + 10,
            "orientation": "fwd",
            "tier": 1,
        }
        for idx, regulator in enumerate(pattern)
    ]


def _write_provenance_fixture(tmp_path: Path) -> dict[str, object]:
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    paths = {
        "opal": tmp_path / "opal" / "records.parquet",
        "latent_rows": tmp_path / "latent" / "rows.parquet",
        "matrix": tmp_path / "latent" / "matrix.npy",
        "dense_source": tmp_path / "densegen" / "records.parquet",
        "dense_sidecar": tmp_path / "densegen" / "_derived" / "densegen.parquet",
        "anchor": tmp_path / "anchor" / "records.parquet",
        "anchor_sidecar": tmp_path / "anchor" / "_derived" / "densegen.parquet",
        "construct_views": tmp_path / "construct" / "_views" / "sequence_views.parquet",
        "feature_aliases": tmp_path / "construct" / "_derived" / "infer" / "feature_aliases.parquet",
        "feature_vectors": tmp_path / "construct" / "_derived" / "infer" / "feature_vectors.parquet",
    }
    for path in paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    _write_fixed_x_records(
        paths["opal"],
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            "source": ["plan_pool__ethanol__sig35_f", "plan_pool__ciprofloxacin__sig35_e"],
            "densegen__plan": [None, "ciprofloxacin__sig35=e"],
            "densegen__run_id": [None, "study_stress_ethanol_cipro"],
            "densegen__sampling_library_hash": [None, "hash-b"],
            "opal_candidate__role": ["opal_candidate_feature_table", "opal_candidate_feature_table"],
            "opal_candidate__x_source_view_id": ["bidir", "bidir"],
            "opal_candidate__source_class": ["densegen", "densegen"],
            "opal_candidate__design_family": ["ethanol", "ciprofloxacin"],
            "opal_candidate__sfxi_ref__collection_id": [None, None],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
        },
        x_col=x_col,
    )
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAAA", "CCCC"],
            "source": ["plan_pool__ethanol__sig35_f", "plan_pool__ciprofloxacin__sig35_e"],
            "densegen__plan": [None, "ciprofloxacin__sig35=e"],
            "densegen__run_id": [None, "study_stress_ethanol_cipro"],
            "densegen__sampling_library_hash": [None, "hash-b"],
        }
    ).to_parquet(paths["dense_source"], index=False)
    pd.DataFrame(
        {
            "id": ["a"],
            "densegen__plan": ["ethanol__sig35=f"],
            "densegen__run_id": ["study_stress_ethanol_cipro"],
            "densegen__sampling_library_hash": ["hash-a"],
        }
    ).to_parquet(paths["dense_sidecar"], index=False)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAAA", "CCCC"],
            "source": ["plan_pool__ethanol__sig35_f", "plan_pool__ciprofloxacin__sig35_e"],
            "densegen__plan": [None, "ciprofloxacin__sig35=e"],
            "densegen__run_id": [None, "study_stress_ethanol_cipro"],
            "densegen__sampling_library_hash": [None, "hash-b"],
        }
    ).to_parquet(paths["anchor"], index=False)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "densegen__plan": ["ethanol__sig35=f", "ciprofloxacin__sig35=e"],
            "densegen__run_id": ["study_stress_ethanol_cipro", "study_stress_ethanol_cipro"],
            "densegen__sampling_library_hash": ["hash-a", "hash-b"],
        }
    ).to_parquet(paths["anchor_sidecar"], index=False)
    pd.DataFrame(
        {
            "construct__anchor_id": ["a", "b"],
            "source_class": ["densegen", "densegen"],
            "design_family": ["ethanol", "ciprofloxacin"],
            "sfxi_ref__collection_id": [None, None],
        }
    ).to_parquet(paths["latent_rows"], index=False)
    np.save(paths["matrix"], np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    pd.DataFrame(
        {
            "view_id": ["view_a_f", "view_a_rc", "view_b_f", "view_b_rc"],
            "sequence_id": ["ctx_a_f", "ctx_a_rc", "ctx_b_f", "ctx_b_rc"],
            "parent_sequence_id": ["a", "a", "b", "b"],
            "parent_dataset_id": ["usr_prom_eth_cip_anchor"] * 4,
            "product_kind": ["realized_context"] * 4,
            "context_kind": ["template_1kb"] * 4,
            "orientation": ["forward", "reverse_complement", "forward", "reverse_complement"],
            "recommended_pooling": ["anchor_mean"] * 4,
        }
    ).to_parquet(paths["construct_views"], index=False)
    pd.DataFrame(
        {
            "alias_id": ["alias_a_f", "alias_a_rc", "alias_b_f", "alias_b_rc"],
            "view_id": ["view_a_f", "view_a_rc", "view_b_f", "view_b_rc"],
            "sequence_id": ["ctx_a_f", "ctx_a_rc", "ctx_b_f", "ctx_b_rc"],
            "feature_vector_key": ["fv_a_f", "fv_a_rc", "fv_b_f", "fv_b_rc"],
            "provider": ["evo2"] * 4,
            "model_name": ["evo2_7b"] * 4,
            "layer_name": ["block26_mlp_out"] * 4,
            "representation_kind": ["intermediate_embedding"] * 4,
            "pooling_operation": ["anchor_mean"] * 4,
            "orientation": ["forward", "reverse_complement", "forward", "reverse_complement"],
            "source_dataset_id": ["construct_prom_eth_cip_context"] * 4,
            "feature_request_digest": ["digest"] * 4,
            "runtime_fingerprint_key": ["runtime"] * 4,
        }
    ).to_parquet(paths["feature_aliases"], index=False)
    pd.DataFrame({"feature_vector_key": ["fv_a_f", "fv_a_rc", "fv_b_f", "fv_b_rc"]}).to_parquet(
        paths["feature_vectors"],
        index=False,
    )

    config = {
        "candidate_feature_table": {
            "records_path": str(paths["opal"].relative_to(tmp_path)),
            "x_column": x_col,
            "x_source": {
                "view_id": "bidir",
                "rows_path": str(paths["latent_rows"].relative_to(tmp_path)),
                "matrix_path": str(paths["matrix"].relative_to(tmp_path)),
            },
            "materialization": {
                "source_records_path": str(paths["anchor"].relative_to(tmp_path)),
                "view_row_id_column": "construct__anchor_id",
                "include_source_class": ["densegen"],
                "allowed_design_families": ["ethanol", "ciprofloxacin"],
                "exclude_non_null_columns": ["sfxi_ref__collection_id"],
            },
        },
        "provenance": {
            "densegen_source_records_path": str(paths["dense_source"].relative_to(tmp_path)),
            "densegen_source_sidecar_path": str(paths["dense_sidecar"].relative_to(tmp_path)),
            "anchor_densegen_sidecar_path": str(paths["anchor_sidecar"].relative_to(tmp_path)),
            "construct_sequence_views_path": str(paths["construct_views"].relative_to(tmp_path)),
            "infer_feature_aliases_path": str(paths["feature_aliases"].relative_to(tmp_path)),
            "infer_feature_vectors_path": str(paths["feature_vectors"].relative_to(tmp_path)),
            "infer_alias_filter": {
                "provider": "evo2",
                "model_name": "evo2_7b",
                "layer_name": "block26_mlp_out",
                "representation_kind": "intermediate_embedding",
                "pooling_operation": "anchor_mean",
            },
        },
    }
    return {"config": config, "x_col": x_col}


def test_select_batch0_enforces_setpoints_and_campaign_slot_splits() -> None:
    config = load_sampling_config(SAMPLING)
    config["synthesis_eligibility"]["min_remaining_candidates"] = 1
    rows = [
        _row(
            "eth_baer_middle",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            sigma="e",
            spacer=17,
            ethanol=0.91,
        ),
        _row(
            "eth_baer_upstream",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "background", "background"),
            sigma="f",
            ethanol=0.90,
        ),
        _row(
            "eth_baer_dense",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "baeR", "baeR"),
            sigma="e",
            spacer=19,
            ethanol=0.89,
        ),
        _row(
            "eth_expl_single",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            sigma="d",
            spacer=18,
            ethanol=0.87,
        ),
        _row(
            "eth_cpxr_middle",
            plan="ethanol",
            regulators="cpxR",
            slot_pattern=("background", "cpxR", "background"),
            sigma="f",
            spacer=17,
            ethanol=0.86,
        ),
        _row(
            "eth_cpxr_upstream",
            plan="ethanol",
            regulators="cpxR",
            slot_pattern=("cpxR", "background", "background"),
            sigma="e",
            ethanol=0.85,
        ),
        _row(
            "cip_lexa_slot0",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("lexA", "background", "background"),
            sigma="f",
            cipro=0.91,
        ),
        _row(
            "cip_lexa_spacer20_high",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("lexA", "background", "background"),
            sigma="f",
            spacer=20,
            cipro=1.5,
        ),
        _row(
            "cip_lexa_slot1",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("background", "lexA", "background"),
            sigma="f",
            cipro=0.90,
        ),
        _row(
            "cip_lexa_slot2",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("background", "background", "lexA"),
            sigma="f",
            cipro=0.89,
        ),
        _row(
            "cip_lexa_dense",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("lexA", "lexA", "lexA"),
            sigma="e",
            spacer=18,
            cipro=0.88,
        ),
        _row(
            "cip_lexa_two_site",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("background", "lexA", "lexA"),
            sigma="f",
            spacer=18,
            cipro=0.87,
        ),
        _row(
            "cip_lexa_expl",
            plan="ciprofloxacin",
            regulators="lexA",
            slot_pattern=("background", "lexA", "background"),
            sigma="d",
            spacer=18,
            cipro=0.86,
        ),
        _row(
            "and_baer_lexa_order_a",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            sigma="f",
            dual=0.96,
        ),
        _row(
            "and_baer_lexa_order_b",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("lexA", "baeR", "background"),
            sigma="f",
            dual=0.95,
        ),
        _row(
            "and_baer_lexa_dense",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("lexA", "baeR", "baeR"),
            sigma="f",
            spacer=18,
            dual=0.94,
        ),
        _row(
            "and_baer_lexa_expl",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            sigma="d",
            spacer=19,
            dual=0.93,
        ),
        _row(
            "and_cpxr_lexa_order_a",
            plan="ethanol_ciprofloxacin",
            regulators="cpxR+lexA",
            slot_pattern=("cpxR", "lexA", "background"),
            sigma="f",
            dual=0.92,
        ),
        _row(
            "and_cpxr_lexa_order_b",
            plan="ethanol_ciprofloxacin",
            regulators="cpxR+lexA",
            slot_pattern=("lexA", "cpxR", "background"),
            sigma="f",
            dual=0.91,
        ),
        _row(
            "and_baer_unclonable_left_junction",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            sigma="f",
            dual=1.25,
            sequence="AATTC" + "A" * 55,
        ),
        _row(
            "negative_prior",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            sigma="f",
            ethanol=-0.1,
        ),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert set(REQUIRED_REVIEW_COLUMNS).issubset(selected.columns)
    assert selected.groupby("campaign").size().to_dict() == {
        "secg_and_rf_sfxi_topn": 6,
        "secg_cipro_rf_sfxi_topn": 6,
        "secg_ethanol_rf_sfxi_topn": 6,
    }
    assert not selected["id"].duplicated().any()
    assert "negative_prior" not in set(selected["id"])
    assert "and_baer_unclonable_left_junction" not in set(selected["id"])
    assert "cip_lexa_spacer20_high" not in set(selected["id"])
    assert "none" not in set(selected["tfbs_summary"])
    assert selected["spacer_length"].isin([16, 17, 18, 19]).all()

    ethanol = selected[selected["campaign"] == "secg_ethanol_rf_sfxi_topn"]
    assert ethanol["setpoint"].map(tuple).unique().tolist() == [(0, 1, 0, 1)]
    assert ethanol["canonical_densegen_plan"].value_counts().to_dict() == {
        "ethanol": 6,
    }
    assert ethanol["regulator_composition"].value_counts().to_dict() == {
        "baeR": 4,
        "cpxR": 2,
    }
    assert ethanol["slot"].tolist() == [
        "ethanol_baer_middle_only",
        "ethanol_baer_upstream_only",
        "ethanol_baer_dense",
        "ethanol_baer_exploratory_low_copy",
        "ethanol_cpxr_middle_only",
        "ethanol_cpxr_upstream_only",
    ]

    cipro = selected[selected["campaign"] == "secg_cipro_rf_sfxi_topn"]
    assert cipro["setpoint"].map(tuple).unique().tolist() == [(0, 0, 1, 1)]
    assert cipro["canonical_densegen_plan"].value_counts().to_dict() == {
        "ciprofloxacin": 6,
    }
    assert cipro["regulator_composition"].value_counts().to_dict() == {
        "lexA": 6,
    }
    assert cipro["slot"].tolist() == [
        "cipro_lexa_slot0",
        "cipro_lexa_slot1",
        "cipro_lexa_slot2",
        "cipro_lexa_dense",
        "cipro_lexa_two_site",
        "cipro_lexa_exploratory_single",
    ]

    and_gate = selected[selected["campaign"] == "secg_and_rf_sfxi_topn"]
    assert and_gate["setpoint"].map(tuple).unique().tolist() == [(0, 0, 0, 1)]
    assert and_gate["canonical_densegen_plan"].unique().tolist() == ["ethanol_ciprofloxacin"]
    assert and_gate["regulator_composition"].value_counts().to_dict() == {
        "baeR+lexA": 4,
        "cpxR+lexA": 2,
    }
    assert and_gate["slot"].tolist() == [
        "and_baer_lexa_baer_before_lexa",
        "and_baer_lexa_lexa_before_baer",
        "and_baer_lexa_dense",
        "and_baer_lexa_exploratory",
        "and_cpxr_lexa_cpxr_before_lexa",
        "and_cpxr_lexa_lexa_before_cpxr",
    ]

    exploratory = selected[selected["slot"].str.contains("exploratory")]
    strong = selected[~selected["slot"].str.contains("exploratory")]
    assert exploratory["sigma35_variant"].isin(["c", "d"]).all()
    assert strong["sigma35_variant"].isin(["f", "e"]).all()
    assert "b" not in set(selected["sigma35_variant"])


def test_select_batch0_supports_exact_slot_patterns_and_signal_tfbs_count() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_ethanol_rf_sfxi_topn",
                "setpoint_vector": [0, 1, 0, 1],
                "target_margin_column": "synthetic_margin_ethanol_vs_background",
                "slots": [
                    {
                        "name": "ethanol_baer_middle_only",
                        "count": 1,
                        "plan": "ethanol",
                        "regulator_compositions": ["baeR"],
                        "slot_regulator_pattern": ["background", "baeR", "background"],
                        "signal_tfbs_count": 1,
                        "allowed_sigma35_variants": ["f", "e"],
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "dense_high_margin_wrong_count",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "baeR", "baeR"),
            ethanol=0.99,
        ),
        _row(
            "upstream_high_margin_wrong_slot",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "background", "background"),
            ethanol=0.98,
        ),
        _row(
            "middle_only_expected",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            ethanol=0.50,
        ),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert selected["id"].tolist() == ["middle_only_expected"]
    assert selected["slot"].tolist() == ["ethanol_baer_middle_only"]


def test_select_batch0_compares_slot_patterns_as_scalar_tuples() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_ethanol_rf_sfxi_topn",
                "setpoint_vector": [0, 1, 0, 1],
                "target_margin_column": "synthetic_margin_ethanol_vs_background",
                "slots": [
                    {
                        "name": "ethanol_baer_middle_only",
                        "count": 1,
                        "plan": "ethanol",
                        "regulator_compositions": ["baeR"],
                        "slot_regulator_pattern": ["background", "baeR", "background"],
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "wrong_dense",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "baeR", "baeR"),
            ethanol=0.99,
        ),
        _row(
            "wrong_upstream",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("baeR", "background", "background"),
            ethanol=0.98,
        ),
        _row(
            "matched_middle",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            ethanol=0.50,
        ),
        _row(
            "wrong_downstream",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "background", "baeR"),
            ethanol=0.97,
        ),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert selected["id"].tolist() == ["matched_middle"]


def test_select_batch0_supports_slot_level_spacer_constraints_and_metadata() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_ethanol_rf_sfxi_topn",
                "setpoint_vector": [0, 1, 0, 1],
                "target_margin_column": "synthetic_margin_ethanol_vs_background",
                "slots": [
                    {
                        "name": "ethanol_baer_middle_only",
                        "count": 1,
                        "plan": "ethanol",
                        "regulator_compositions": ["baeR"],
                        "slot_regulator_pattern": ["background", "baeR", "background"],
                        "signal_tfbs_count": 1,
                        "allowed_sigma35_variants": ["f", "e"],
                        "allowed_spacer_lengths": [17],
                        "design_hypothesis": "BaeR middle slot tunes -35-side RNAP geometry.",
                        "primary_comparison": "ethanol_baer_upstream_only",
                        "geometry_hypothesis": "functional_realignment",
                        "interpretation_limit": "Fixed TATAAT -10; not a -10-strength test.",
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "wrong_spacer_high_margin",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            spacer=16,
            ethanol=0.99,
        ),
        _row(
            "matched_spacer",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            spacer=17,
            ethanol=0.50,
        ),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert selected["id"].tolist() == ["matched_spacer"]
    assert selected["spacer_length"].tolist() == [17]
    assert selected["design_hypothesis"].tolist() == ["BaeR middle slot tunes -35-side RNAP geometry."]
    assert selected["primary_comparison"].tolist() == ["ethanol_baer_upstream_only"]
    assert selected["geometry_hypothesis"].tolist() == ["functional_realignment"]
    assert selected["interpretation_limit"].tolist() == ["Fixed TATAAT -10; not a -10-strength test."]


def test_select_batch0_fails_when_slot_level_spacer_constraint_exhausts_pool() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_ethanol_rf_sfxi_topn",
                "setpoint_vector": [0, 1, 0, 1],
                "target_margin_column": "synthetic_margin_ethanol_vs_background",
                "slots": [
                    {
                        "name": "ethanol_baer_middle_only",
                        "count": 1,
                        "plan": "ethanol",
                        "regulator_compositions": ["baeR"],
                        "slot_regulator_pattern": ["background", "baeR", "background"],
                        "signal_tfbs_count": 1,
                        "allowed_spacer_lengths": [19],
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "only_wrong_spacer",
            plan="ethanol",
            regulators="baeR",
            slot_pattern=("background", "baeR", "background"),
            spacer=17,
            ethanol=0.9,
        )
    ]

    with pytest.raises(ValueError, match="requires 1 rows but only 0 candidates passed filters"):
        select_batch0(pd.DataFrame(rows), config)


def test_select_batch0_applies_off_target_margin_constraints() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_and_rf_sfxi_topn",
                "setpoint_vector": [0, 0, 0, 1],
                "target_margin_column": "synthetic_margin_dual_vs_background",
                "off_target_margin_columns": [
                    "synthetic_margin_ethanol_vs_background",
                    "synthetic_margin_cipro_vs_background",
                ],
                "off_target_margin_constraints": {"max_each": 0.3},
                "slots": [
                    {
                        "name": "and_baer_lexa_baer_before_lexa",
                        "count": 1,
                        "plan": "ethanol_ciprofloxacin",
                        "regulator_compositions": ["baeR+lexA"],
                        "slot_regulator_pattern": ["baeR", "lexA", "background"],
                        "signal_tfbs_count": 2,
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "generic_dual_positive_high_margin",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            ethanol=0.70,
            cipro=0.65,
            dual=0.95,
        ),
        _row(
            "and_specific_lower_margin",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            ethanol=0.20,
            cipro=0.25,
            dual=0.55,
        ),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert selected["id"].tolist() == ["and_specific_lower_margin"]


def test_select_batch0_applies_campaign_target_margin_minimum() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_and_rf_sfxi_topn",
                "setpoint_vector": [0, 0, 0, 1],
                "target_margin_column": "synthetic_margin_dual_vs_background",
                "target_margin_min": 0.2,
                "slots": [
                    {
                        "name": "and_baer_lexa_baer_before_lexa",
                        "count": 1,
                        "plan": "ethanol_ciprofloxacin",
                        "regulator_compositions": ["baeR+lexA"],
                        "slot_regulator_pattern": ["baeR", "lexA", "background"],
                        "signal_tfbs_count": 2,
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    rows = [
        _row(
            "barely_positive_dual_margin",
            plan="ethanol_ciprofloxacin",
            regulators="baeR+lexA",
            slot_pattern=("baeR", "lexA", "background"),
            dual=0.05,
        )
    ]

    with pytest.raises(ValueError, match="requires 1 rows but only 0 candidates passed filters"):
        select_batch0(pd.DataFrame(rows), config)


def test_select_batch0_fails_fast_when_slot_predicate_cannot_parse_densegen_detail() -> None:
    config = {
        "allow_duplicate_ids": False,
        "campaigns": [
            {
                "slug": "secg_cipro_rf_sfxi_topn",
                "setpoint_vector": [0, 0, 1, 1],
                "target_margin_column": "synthetic_margin_cipro_vs_background",
                "slots": [
                    {
                        "name": "cipro_lexa_slot0",
                        "count": 1,
                        "plan": "ciprofloxacin",
                        "regulator_compositions": ["lexA"],
                        "slot_regulator_pattern": ["lexA", "background", "background"],
                    }
                ],
            }
        ],
        "filters": {"require_positive_target_margin": True},
    }
    row = _row("missing_detail", plan="ciprofloxacin", regulators="lexA", cipro=0.9)
    row.pop("densegen__used_tfbs_detail", None)

    with pytest.raises(ValueError, match="missing_detail: missing densegen__used_tfbs_detail"):
        select_batch0(pd.DataFrame([row]), config)

    malformed = _row(
        "two_slot_detail",
        plan="ciprofloxacin",
        regulators="lexA",
        slot_pattern=("lexA", "background", "background"),
        cipro=0.9,
    )
    malformed["densegen__used_tfbs_detail"] = malformed["densegen__used_tfbs_detail"][:2]  # type: ignore[index]

    with pytest.raises(ValueError, match="two_slot_detail: expected exactly 3 TFBS entries"):
        select_batch0(pd.DataFrame([malformed]), config)


def test_configured_sampling_yaml_declares_granular_batch0_composition() -> None:
    config = load_sampling_config(SAMPLING)
    campaigns = {campaign["objective"]: campaign for campaign in config["campaigns"]}
    required_metadata = {
        "design_hypothesis",
        "primary_comparison",
        "geometry_hypothesis",
        "interpretation_limit",
    }

    ethanol_slots = campaigns["ethanol_factor"]["slots"]
    assert all(required_metadata.issubset(slot) for slot in ethanol_slots)
    assert [slot["name"] for slot in ethanol_slots] == [
        "ethanol_baer_middle_only",
        "ethanol_baer_upstream_only",
        "ethanol_baer_dense",
        "ethanol_baer_exploratory_low_copy",
        "ethanol_cpxr_middle_only",
        "ethanol_cpxr_upstream_only",
    ]
    assert [slot["regulator_compositions"] for slot in ethanol_slots] == [
        ["baeR"],
        ["baeR"],
        ["baeR"],
        ["baeR"],
        ["cpxR"],
        ["cpxR"],
    ]
    assert all(slot["plan"] == "ethanol" for slot in ethanol_slots)
    assert not any("lexA" in "+".join(slot["regulator_compositions"]) for slot in ethanol_slots)
    assert ethanol_slots[0]["slot_regulator_pattern"] == ["background", "baeR", "background"]
    assert ethanol_slots[0]["allowed_spacer_lengths"] == [17]
    assert ethanol_slots[1]["slot_regulator_pattern"] == ["baeR", "background", "background"]
    assert ethanol_slots[1]["allowed_spacer_lengths"] == [16]
    assert ethanol_slots[2]["slot_regulator_pattern"] == ["baeR", "baeR", "baeR"]
    assert ethanol_slots[2]["signal_tfbs_count"] == 3
    assert ethanol_slots[2]["allowed_spacer_lengths"] == [19]
    assert ethanol_slots[3]["slot_regulator_pattern"] == ["background", "baeR", "background"]
    assert ethanol_slots[3]["signal_tfbs_count"] == 1
    assert ethanol_slots[3]["allowed_spacer_lengths"] == [18]
    assert ethanol_slots[4]["allowed_spacer_lengths"] == [17]
    assert ethanol_slots[5]["allowed_spacer_lengths"] == [16]

    cipro_slots = campaigns["cipro_factor"]["slots"]
    assert all(required_metadata.issubset(slot) for slot in cipro_slots)
    assert [slot["name"] for slot in cipro_slots] == [
        "cipro_lexa_slot0",
        "cipro_lexa_slot1",
        "cipro_lexa_slot2",
        "cipro_lexa_dense",
        "cipro_lexa_two_site",
        "cipro_lexa_exploratory_single",
    ]
    assert all(slot["plan"] == "ciprofloxacin" for slot in cipro_slots)
    assert all(slot["regulator_compositions"] == ["lexA"] for slot in cipro_slots)
    assert cipro_slots[0]["slot_regulator_pattern"] == ["lexA", "background", "background"]
    assert cipro_slots[1]["slot_regulator_pattern"] == ["background", "lexA", "background"]
    assert cipro_slots[2]["slot_regulator_pattern"] == ["background", "background", "lexA"]
    assert [slot["allowed_spacer_lengths"] for slot in cipro_slots[:3]] == [[16], [16], [16]]
    assert all(slot["allowed_sigma35_variants"] == ["f"] for slot in cipro_slots[:3])
    assert cipro_slots[3]["slot_regulator_pattern"] == ["lexA", "lexA", "lexA"]
    assert cipro_slots[3]["signal_tfbs_count"] == 3
    assert cipro_slots[3]["allowed_spacer_lengths"] == [18]
    assert cipro_slots[4]["slot_regulator_pattern"] == ["background", "lexA", "lexA"]
    assert cipro_slots[4]["signal_tfbs_count"] == 2
    assert cipro_slots[4]["allowed_spacer_lengths"] == [18]
    assert cipro_slots[5]["slot_regulator_pattern"] == ["background", "lexA", "background"]
    assert cipro_slots[5]["signal_tfbs_count"] == 1
    assert cipro_slots[5]["allowed_spacer_lengths"] == [18]

    and_slots = campaigns["and"]["slots"]
    assert campaigns["and"]["target_margin_min"] == 0.2
    assert campaigns["and"]["off_target_margin_constraints"] == {"max_each": 0.3}
    assert all(required_metadata.issubset(slot) for slot in and_slots)
    assert [slot["name"] for slot in and_slots] == [
        "and_baer_lexa_baer_before_lexa",
        "and_baer_lexa_lexa_before_baer",
        "and_baer_lexa_dense",
        "and_baer_lexa_exploratory",
        "and_cpxr_lexa_cpxr_before_lexa",
        "and_cpxr_lexa_lexa_before_cpxr",
    ]
    assert [slot["regulator_compositions"] for slot in and_slots] == [
        ["baeR+lexA"],
        ["baeR+lexA"],
        ["baeR+lexA"],
        ["baeR+lexA"],
        ["cpxR+lexA"],
        ["cpxR+lexA"],
    ]
    assert and_slots[0]["slot_regulator_pattern"] == ["baeR", "lexA", "background"]
    assert and_slots[1]["slot_regulator_pattern"] == ["lexA", "baeR", "background"]
    assert and_slots[0]["allowed_spacer_lengths"] == [16]
    assert and_slots[1]["allowed_spacer_lengths"] == [16]
    assert and_slots[2]["slot_regulator_pattern"] == ["lexA", "baeR", "baeR"]
    assert and_slots[2]["signal_tfbs_count"] == 3
    assert and_slots[2]["allowed_spacer_lengths"] == [18]
    assert and_slots[3]["slot_regulator_pattern"] == ["baeR", "lexA", "background"]
    assert and_slots[3]["signal_tfbs_count"] == 2
    assert and_slots[3]["allowed_spacer_lengths"] == [19]
    assert and_slots[4]["slot_regulator_pattern"] == ["cpxR", "lexA", "background"]
    assert and_slots[5]["slot_regulator_pattern"] == ["lexA", "cpxR", "background"]
    assert and_slots[4]["allowed_spacer_lengths"] == [16]
    assert and_slots[5]["allowed_spacer_lengths"] == [16]


def test_candidate_feature_table_validation_requires_fixed_length_x_and_view_alignment(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.parquet"
    view_rows = tmp_path / "view_rows.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        },
        x_col=x_col,
    )
    pd.DataFrame({"construct__anchor_id": ["a", "b"]}).to_parquet(view_rows)

    report = validate_candidate_feature_table(
        records_path=records,
        x_column=x_col,
        view_rows_path=view_rows,
        view_row_id_column="construct__anchor_id",
    )

    assert report["row_count"] == 2
    assert report["x_dim"] == 2

    pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4, 0.5]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        }
    ).to_parquet(records)

    with pytest.raises(ValueError, match="fixed_size_list"):
        validate_candidate_feature_table(records_path=records, x_column=x_col)


def test_candidate_feature_table_validation_does_not_pandas_load_x(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tmp_path / "records.parquet"
    view_rows = tmp_path / "view_rows.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        },
        x_col=x_col,
    )
    pd.DataFrame({"construct__anchor_id": ["a", "b"]}).to_parquet(view_rows)

    calls: list[tuple[str, ...] | None] = []
    original = pd.read_parquet

    def spy_read_parquet(path, *args, columns=None, **kwargs):
        calls.append(tuple(columns) if columns is not None else None)
        assert columns is not None
        assert x_col not in columns
        return original(path, *args, columns=columns, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy_read_parquet)

    report = validate_candidate_feature_table(
        records_path=records,
        x_column=x_col,
        view_rows_path=view_rows,
        view_row_id_column="construct__anchor_id",
    )

    assert report == {
        "row_count": 2,
        "x_dim": 2,
        "densegen_metadata_row_count": 2,
        "densegen_metadata_exempt_row_count": 0,
    }
    assert calls == [
        (
            "id",
            "bio_type",
            "sequence",
            "alphabet",
            "opal_candidate__role",
            "opal_candidate__x_source_view_id",
            "opal_candidate__source_class",
            "opal_candidate__design_family",
            "opal_candidate__sfxi_ref__collection_id",
            "densegen__plan",
            "densegen__run_id",
            "densegen__sampling_library_hash",
        ),
        ("construct__anchor_id",),
    ]


def test_candidate_feature_table_validation_enforces_exact_population_contract(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b", "c"],
            "bio_type": ["dna", "dna", "dna"],
            "sequence": ["AAAA", "CCCC", "GGGG"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            **_candidate_provenance(
                3,
                source_classes=["densegen", "densegen", "native"],
                design_families=["ethanol", "ciprofloxacin", "control"],
            ),
        },
        x_col=x_col,
    )

    with pytest.raises(ValueError, match="row count 3 does not equal expected 2"):
        validate_candidate_feature_table(records_path=records, x_column=x_col, expected_rows=2)

    with pytest.raises(ValueError, match="opal_candidate__source_class"):
        validate_candidate_feature_table(
            records_path=records,
            x_column=x_col,
            expected_rows=3,
            allowed_source_classes=["densegen"],
        )

    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "control"]),
        },
        x_col=x_col,
    )
    with pytest.raises(ValueError, match="opal_candidate__design_family"):
        validate_candidate_feature_table(
            records_path=records,
            x_column=x_col,
            allowed_design_families=["ethanol", "ciprofloxacin"],
        )

    provenance = _candidate_provenance(1)
    provenance["opal_candidate__sfxi_ref__collection_id"] = ["archive-sfxi"]
    _write_fixed_x_records(
        records,
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
            **provenance,
        },
        x_col=x_col,
    )
    with pytest.raises(ValueError, match="opal_candidate__sfxi_ref__collection_id"):
        validate_candidate_feature_table(
            records_path=records,
            x_column=x_col,
            required_null_provenance_columns=["opal_candidate__sfxi_ref__collection_id"],
        )


def test_candidate_feature_table_validation_allows_ordered_latentdna_view_subset(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    view_rows = tmp_path / "view_rows.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    _write_fixed_x_records(
        records,
        {
            "id": ["a", "c"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        },
        x_col=x_col,
    )
    pd.DataFrame({"construct__anchor_id": ["a", "b", "c"]}).to_parquet(view_rows)

    assert validate_candidate_feature_table(
        records_path=records,
        x_column=x_col,
        view_rows_path=view_rows,
        view_row_id_column="construct__anchor_id",
    ) == {
        "row_count": 2,
        "x_dim": 2,
        "densegen_metadata_row_count": 2,
        "densegen_metadata_exempt_row_count": 0,
    }


def test_candidate_feature_table_validation_fails_fast_when_records_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate feature table records_path not found"):
        validate_candidate_feature_table(
            records_path=tmp_path / "missing" / "records.parquet",
            x_column="latentdna__evo2_7b__context_anchor_mean_bidir_concat",
        )


def test_candidate_feature_table_validation_rejects_null_required_opal_values(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b"],
            "bio_type": ["dna", None],
            "sequence": ["AAAA", ""],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        },
        x_col=x_col,
    )

    with pytest.raises(ValueError, match="required column 'bio_type' has null/blank values"):
        validate_candidate_feature_table(records_path=records, x_column=x_col)


def test_candidate_feature_table_validation_requires_study_provenance_columns(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
        },
        x_col=x_col,
    )

    with pytest.raises(ValueError, match="opal_candidate__role"):
        validate_candidate_feature_table(records_path=records, x_column=x_col)


def test_candidate_feature_table_validation_rejects_blank_core_provenance(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    provenance = _candidate_provenance(1)
    provenance["opal_candidate__source_class"] = [""]
    _write_fixed_x_records(
        records,
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
            **provenance,
        },
        x_col=x_col,
    )

    with pytest.raises(ValueError, match="opal_candidate__source_class"):
        validate_candidate_feature_table(records_path=records, x_column=x_col)


def test_configured_candidate_feature_table_validation_resolves_repo_paths(tmp_path: Path) -> None:
    records = tmp_path / "usr" / "datasets" / "demo" / "records.parquet"
    view_rows = tmp_path / "latentdna" / "views" / "rows.parquet"
    records.parent.mkdir(parents=True)
    view_rows.parent.mkdir(parents=True)
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
            **_candidate_provenance(1),
        },
        x_col=x_col,
    )
    pd.DataFrame({"construct__anchor_id": ["a"]}).to_parquet(view_rows)

    report = validate_configured_candidate_feature_table(
        {
            "candidate_feature_table": {
                "records_path": "usr/datasets/demo/records.parquet",
                "x_column": x_col,
                "x_source": {"rows_path": "latentdna/views/rows.parquet"},
            }
        },
        repo_root=tmp_path,
    )

    assert report == {
        "row_count": 1,
        "x_dim": 2,
        "densegen_metadata_row_count": 1,
        "densegen_metadata_exempt_row_count": 0,
    }


def test_batch0_preview_fails_fast_when_candidate_table_view_rows_drift(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    view_rows = tmp_path / "view_rows.parquet"
    config_path = tmp_path / "sampling.yaml"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            **_candidate_provenance(2, design_families=["ethanol", "ciprofloxacin"]),
        },
        x_col=x_col,
    )
    pd.DataFrame({"construct__anchor_id": ["b", "a"]}).to_parquet(view_rows)
    config_path.write_text(
        yaml.safe_dump(
            {
                "campaigns": [],
                "candidate_feature_table": {
                    "records_path": "records.parquet",
                    "x_column": x_col,
                    "x_source": {"rows_path": "view_rows.parquet"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="align with LatentDNA view rows"):
        batch0_select_main(["--config", str(config_path), "--repo-root", str(tmp_path)])


def test_candidate_table_materializer_filters_dense_plan_subset_and_writes_x(tmp_path: Path) -> None:
    rows_path = tmp_path / "view_rows.parquet"
    matrix_path = tmp_path / "matrix.npy"
    source_path = tmp_path / "anchor" / "records.parquet"
    densegen_sidecar_path = tmp_path / "anchor" / "_derived" / "densegen.parquet"
    records_path = tmp_path / "opal" / "records.parquet"
    config_path = tmp_path / "sampling.yaml"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    pd.DataFrame(
        {
            "construct__anchor_id": ["a", "b", "c", "d", "e"],
            "source_class": ["densegen", "densegen", "manual_or_wildtype", "densegen", "densegen"],
            "design_family": ["ethanol", "ciprofloxacin", "control", "control", "ethanol"],
            "sfxi_ref__collection_id": [None, None, None, None, "reader_sfxi_pdual10_latest"],
        }
    ).to_parquet(rows_path)
    np.save(matrix_path, np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32))
    source_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "bio_type": ["dna"] * 5,
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT", "ACAC"],
            "alphabet": ["dna_4"] * 5,
            "densegen__plan": [None, "ciprofloxacin", "control", "control", "ethanol"],
            "densegen__run_id": [None, "run-b", "run-c", "run-d", "run-e"],
            "densegen__sampling_library_hash": [None, "hash-b", "hash-c", "hash-d", "hash-e"],
        }
    ).to_parquet(source_path, index=False)
    densegen_sidecar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "densegen__plan": ["ethanol", "ciprofloxacin"],
            "densegen__run_id": ["run-a", "run-b"],
            "densegen__sampling_library_hash": ["hash-a", "hash-b"],
            "densegen__used_tfbs_detail": [
                _densegen_tfbs_detail(("background", "baeR", "background")),
                _densegen_tfbs_detail(("background", "lexA", "background")),
            ],
            "densegen__required_regulators": [["baeR"], ["lexA"]],
        }
    ).to_parquet(densegen_sidecar_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "campaigns": [],
                "candidate_feature_table": {
                    "dataset_id": "demo_opal_candidates",
                    "role": "opal_candidate_feature_table",
                    "records_path": "opal/records.parquet",
                    "x_column": x_col,
                    "x_source": {
                        "view_id": "bidir_context",
                        "rows_path": "view_rows.parquet",
                        "matrix_path": "matrix.npy",
                    },
                    "materialization": {
                        "source_records_path": "anchor/records.parquet",
                        "densegen_sidecar_path": "anchor/_derived/densegen.parquet",
                        "densegen_sidecar_columns": [
                            "densegen__plan",
                            "densegen__run_id",
                            "densegen__sampling_library_hash",
                            "densegen__used_tfbs_detail",
                            "densegen__required_regulators",
                        ],
                        "view_row_id_column": "construct__anchor_id",
                        "include_source_class": ["densegen"],
                        "allowed_design_families": ["ethanol", "ciprofloxacin"],
                        "exclude_non_null_columns": ["sfxi_ref__collection_id"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_sampling_config(config_path)

    dry_run = materialize_configured_candidate_feature_table(config, repo_root=tmp_path, write=False)
    assert dry_run["row_count"] == 2
    assert dry_run["x_dim"] == 2
    assert not records_path.exists()

    assert candidate_table_main(["--config", str(config_path), "--repo-root", str(tmp_path), "--write"]) == 0
    assert (
        candidate_table_main(["--config", str(config_path), "--repo-root", str(tmp_path), "--validate-existing"]) == 0
    )
    records = pd.read_parquet(records_path)
    assert records["id"].tolist() == ["a", "b"]
    assert records[x_col].map(list).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert records["densegen__plan"].tolist() == ["ethanol", "ciprofloxacin"]
    assert records["densegen__run_id"].tolist() == ["run-a", "run-b"]
    assert records["densegen__sampling_library_hash"].tolist() == ["hash-a", "hash-b"]
    assert set(
        [
            "opal_candidate__role",
            "opal_candidate__x_source_view_id",
            "opal_candidate__source_class",
            "opal_candidate__design_family",
            "opal_candidate__sfxi_ref__collection_id",
        ]
    ).issubset(records.columns)
    assert records["opal_candidate__source_class"].tolist() == ["densegen", "densegen"]
    assert records["opal_candidate__design_family"].tolist() == ["ethanol", "ciprofloxacin"]
    assert records["opal_candidate__sfxi_ref__collection_id"].isna().all()


def test_candidate_table_materializer_manual_includes_measured_reader_rows(tmp_path: Path) -> None:
    rows_path = tmp_path / "view_rows.parquet"
    matrix_path = tmp_path / "matrix.npy"
    source_path = tmp_path / "anchor" / "records.parquet"
    densegen_sidecar_path = tmp_path / "anchor" / "_derived" / "densegen.parquet"
    records_path = tmp_path / "opal" / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    pd.DataFrame(
        {
            "construct__anchor_id": ["dense-a", "sfxi-b", "control-c"],
            "source_class": ["densegen", "densegen", "construct_derived"],
            "design_family": ["ethanol", "ethanol_ciprofloxacin", "control"],
            "sfxi_ref__collection_id": [None, "reader_sfxi_pdual10_latest", None],
        }
    ).to_parquet(rows_path)
    np.save(matrix_path, np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
    source_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": ["dense-a", "sfxi-b", "control-c"],
            "bio_type": ["dna"] * 3,
            "sequence": ["AAAA", "CCCC", "GGGG"],
            "alphabet": ["dna_4"] * 3,
            "densegen__plan": ["ethanol", None, None],
            "densegen__run_id": ["run-a", None, None],
            "densegen__sampling_library_hash": ["hash-a", None, None],
        }
    ).to_parquet(source_path, index=False)
    densegen_sidecar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["dense-a", "sfxi-b"],
            "densegen__plan": ["ethanol", "ethanol_ciprofloxacin"],
            "densegen__run_id": ["run-a", "reader_sfxi_archive"],
            "densegen__sampling_library_hash": ["hash-a", "archive-hash"],
            "densegen__used_tfbs_detail": [
                _densegen_tfbs_detail(("background", "baeR", "background")),
                _densegen_tfbs_detail(("baeR", "lexA", "background")),
            ],
            "densegen__required_regulators": [["baeR"], ["baeR", "lexA"]],
        }
    ).to_parquet(densegen_sidecar_path, index=False)
    config = {
        "candidate_feature_table": {
            "dataset_id": "demo_opal_candidates",
            "role": "opal_candidate_feature_table",
            "records_path": "opal/records.parquet",
            "expected_rows": 3,
            "x_column": x_col,
            "x_source": {
                "view_id": "bidir_context",
                "rows_path": "view_rows.parquet",
                "matrix_path": "matrix.npy",
            },
            "materialization": {
                "source_records_path": "anchor/records.parquet",
                "densegen_sidecar_path": "anchor/_derived/densegen.parquet",
                "densegen_sidecar_columns": [
                    "densegen__plan",
                    "densegen__run_id",
                    "densegen__sampling_library_hash",
                    "densegen__used_tfbs_detail",
                    "densegen__required_regulators",
                ],
                "view_row_id_column": "construct__anchor_id",
                "include_source_class": ["densegen"],
                "allowed_design_families": ["ethanol"],
                "exclude_non_null_columns": ["sfxi_ref__collection_id"],
                "manual_include_view_row_ids": ["sfxi-b", "control-c"],
                "allow_missing_densegen_sidecar_for_non_densegen": True,
                "validation_allowed_source_classes": ["densegen", "construct_derived"],
                "validation_allowed_design_families": ["ethanol", "ethanol_ciprofloxacin", "control"],
                "validation_required_null_provenance_columns": [],
            },
        }
    }

    report = materialize_configured_candidate_feature_table(config, repo_root=tmp_path, write=True)

    assert report["row_count"] == 3
    assert report["validation"] == {
        "row_count": 3,
        "x_dim": 2,
        "densegen_metadata_row_count": 2,
        "densegen_metadata_exempt_row_count": 1,
    }
    records = pd.read_parquet(records_path)
    assert records["id"].tolist() == ["dense-a", "sfxi-b", "control-c"]
    assert records[x_col].map(list).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert records["densegen__plan"].iloc[:2].tolist() == ["ethanol", "ethanol_ciprofloxacin"]
    assert pd.isna(records["densegen__plan"].iloc[2])
    assert records["opal_candidate__source_class"].tolist() == ["densegen", "densegen", "construct_derived"]
    assert pd.isna(records["opal_candidate__sfxi_ref__collection_id"].iloc[0])
    assert records["opal_candidate__sfxi_ref__collection_id"].iloc[1] == "reader_sfxi_pdual10_latest"
    assert pd.isna(records["opal_candidate__sfxi_ref__collection_id"].iloc[2])


def test_candidate_table_write_reuses_configured_validation_guards(tmp_path: Path) -> None:
    rows_path = tmp_path / "view_rows.parquet"
    matrix_path = tmp_path / "matrix.npy"
    source_path = tmp_path / "anchor" / "records.parquet"
    densegen_sidecar_path = tmp_path / "anchor" / "_derived" / "densegen.parquet"
    config_path = tmp_path / "sampling.yaml"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    pd.DataFrame(
        {
            "construct__anchor_id": ["a", "b"],
            "source_class": ["densegen", "densegen"],
            "design_family": ["ethanol", "ciprofloxacin"],
            "sfxi_ref__collection_id": [None, None],
        }
    ).to_parquet(rows_path)
    np.save(matrix_path, np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    source_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            "densegen__plan": ["ethanol", "ciprofloxacin"],
            "densegen__run_id": ["run-a", "run-b"],
            "densegen__sampling_library_hash": ["hash-a", "hash-b"],
            "densegen__used_tfbs_detail": [
                _densegen_tfbs_detail(("background", "baeR", "background")),
                _densegen_tfbs_detail(("background", "lexA", "background")),
            ],
            "densegen__required_regulators": [["baeR"], ["lexA"]],
        }
    ).to_parquet(source_path, index=False)
    densegen_sidecar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "densegen__plan": ["ethanol", "ciprofloxacin"],
            "densegen__run_id": ["run-a", "run-b"],
            "densegen__sampling_library_hash": ["hash-a", "hash-b"],
            "densegen__used_tfbs_detail": [
                _densegen_tfbs_detail(("background", "baeR", "background")),
                _densegen_tfbs_detail(("background", "lexA", "background")),
            ],
            "densegen__required_regulators": [["baeR"], ["lexA"]],
        }
    ).to_parquet(densegen_sidecar_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "campaigns": [],
                "candidate_feature_table": {
                    "records_path": "opal/records.parquet",
                    "expected_rows": 3,
                    "x_column": x_col,
                    "x_source": {
                        "rows_path": "view_rows.parquet",
                        "matrix_path": "matrix.npy",
                    },
                    "materialization": {
                        "source_records_path": "anchor/records.parquet",
                        "densegen_sidecar_path": "anchor/_derived/densegen.parquet",
                        "densegen_sidecar_columns": [
                            "densegen__plan",
                            "densegen__run_id",
                            "densegen__sampling_library_hash",
                            "densegen__used_tfbs_detail",
                            "densegen__required_regulators",
                        ],
                        "view_row_id_column": "construct__anchor_id",
                        "include_source_class": ["densegen"],
                        "allowed_design_families": ["ethanol", "ciprofloxacin"],
                        "exclude_non_null_columns": ["sfxi_ref__collection_id"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count 2 does not equal expected 3"):
        candidate_table_main(["--config", str(config_path), "--repo-root", str(tmp_path), "--write"])


def test_candidate_table_materializer_fails_fast_on_incomplete_densegen_sidecar(tmp_path: Path) -> None:
    rows_path = tmp_path / "view_rows.parquet"
    matrix_path = tmp_path / "matrix.npy"
    source_path = tmp_path / "anchor" / "records.parquet"
    densegen_sidecar_path = tmp_path / "anchor" / "_derived" / "densegen.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    pd.DataFrame(
        {
            "construct__anchor_id": ["a", "b"],
            "source_class": ["densegen", "densegen"],
            "design_family": ["ethanol", "ciprofloxacin"],
        }
    ).to_parquet(rows_path)
    np.save(matrix_path, np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    source_path.parent.mkdir()
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
        }
    ).to_parquet(source_path, index=False)
    densegen_sidecar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a"],
            "densegen__plan": ["ethanol"],
            "densegen__run_id": ["run-a"],
            "densegen__sampling_library_hash": ["hash-a"],
        }
    ).to_parquet(densegen_sidecar_path, index=False)

    config = {
        "candidate_feature_table": {
            "records_path": "opal/records.parquet",
            "x_column": x_col,
            "x_source": {"rows_path": "view_rows.parquet", "matrix_path": "matrix.npy"},
            "materialization": {
                "source_records_path": "anchor/records.parquet",
                "densegen_sidecar_path": "anchor/_derived/densegen.parquet",
                "densegen_sidecar_columns": [
                    "densegen__plan",
                    "densegen__run_id",
                    "densegen__sampling_library_hash",
                ],
                "view_row_id_column": "construct__anchor_id",
                "include_source_class": ["densegen"],
                "allowed_design_families": ["ethanol", "ciprofloxacin"],
            },
        }
    }

    with pytest.raises(ValueError, match="missing from DenseGen sidecar"):
        materialize_configured_candidate_feature_table(config, repo_root=tmp_path, write=False)

    pd.DataFrame(
        {
            "id": ["a", "b", "b"],
            "densegen__plan": ["ethanol", "ciprofloxacin", "ciprofloxacin"],
            "densegen__run_id": ["run-a", "run-b", "run-b"],
            "densegen__sampling_library_hash": ["hash-a", "hash-b", "hash-b"],
        }
    ).to_parquet(densegen_sidecar_path, index=False)
    with pytest.raises(ValueError, match="duplicate ids"):
        materialize_configured_candidate_feature_table(config, repo_root=tmp_path, write=False)


def test_candidate_lineage_audit_uses_sidecars_without_loading_x_payload(tmp_path: Path) -> None:
    fixture = _write_provenance_fixture(tmp_path)
    config = fixture["config"]

    report = audit_candidate_lineage(config, repo_root=tmp_path)

    assert report["status"] == "pass_with_sidecar"
    assert report["attention"] == []
    assert report["candidate_table"]["row_count"] == 2
    assert report["identity"]["ids_equal_densegen_source"] is True
    assert report["identity"]["sequence_mismatch_vs_densegen_source"] == 0
    assert report["latentdna"]["selected_order_matches_opal"] is True
    assert report["construct"]["matched_view_rows"] == 4
    assert report["construct"]["ids_with_not_two_views"] == 0
    assert report["infer"]["matched_alias_rows"] == 4
    assert report["infer"]["missing_feature_vector_keys"] == 0
    assert report["densegen_metadata"]["opal_record_null_counts"]["densegen__plan"] == 1
    assert report["densegen_metadata"]["anchor_densegen_sidecar_missing_opal_ids"] == 0
    assert report["densegen_metadata"]["resolution_state"] == "complete_via_anchor_sidecar"


def test_candidate_lineage_show_resolves_densegen_provenance_from_anchor_sidecar(tmp_path: Path) -> None:
    fixture = _write_provenance_fixture(tmp_path)
    config = fixture["config"]

    report = show_candidate_lineage(config, repo_root=tmp_path, candidate_id="a")

    assert report["id"] == "a"
    assert report["sequence"] == "AAAA"
    assert report["row_positions"] == {"opal_candidate_records": 0, "latentdna_rows": 0}
    assert report["opal"]["x_payload_loaded"] is False
    assert report["densegen"]["resolved"]["densegen__plan"] == "ethanol__sig35=f"
    assert report["densegen"]["resolved_from"]["densegen__plan"] == "anchor_densegen_sidecar"
    assert report["latentdna"]["derive_kind"] == "block_normalized_concatenate"
    assert {row["orientation"] for row in report["construct"]["views"]} == {"forward", "reverse_complement"}
    assert {row["orientation"] for row in report["infer"]["aliases"]} == {"forward", "reverse_complement"}
    assert report["warnings"] == []


def test_selected_ids_must_exist_in_candidate_feature_table(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    _write_fixed_x_records(
        records,
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
        },
        x_col=x_col,
    )

    with pytest.raises(ValueError, match="missing from the OPAL candidate feature table"):
        validate_selected_ids_against_candidate_feature_table(
            pd.DataFrame({"id": ["b"]}),
            {"candidate_feature_table": {"records_path": "records.parquet"}},
            repo_root=tmp_path,
        )


def test_opal_campaign_configs_point_at_candidate_feature_table() -> None:
    expected = {
        "ethanol": [0, 1, 0, 1],
        "ciprofloxacin": [0, 0, 1, 1],
        "and": [0, 0, 0, 1],
    }
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    cfg = load_config(CAMPAIGN_ROOT / "secg_msrb_greedy" / "configs/campaign.yaml")
    assert cfg.campaign.slug == "secg_msrb_greedy"
    assert cfg.data.location.kind == "usr"
    assert cfg.data.location.dataset == "usr_prom_eth_cip_opal_candidates"
    assert cfg.data.location.path.endswith("src/dnadesign/usr/datasets")
    assert cfg.data.x_column_name == x_col
    assert cfg.data.y_column_name == "opal__reader_response_window_vector_v1__y"
    assert cfg.data.y_expected_length == 8
    assert cfg.labels.source.kind == "usr_sidecar"
    assert cfg.labels.source.dataset == "usr_prom_eth_cip_opal_candidates"
    assert cfg.labels.source.path == "_opal/response_window_labels_v5/observed_labels.parquet"
    assert cfg.labels.source.manifest_path == "_opal/response_window_labels_v5/promotion.manifest.json"
    assert cfg.labels.y_space == "reader_response_window_vector_v1"
    assert cfg.labels.round_column == "observed_round"
    assert cfg.labels.dedup_policy == "error_on_duplicate"
    assert cfg.writeback.prediction_records == "ledger_only"
    assert cfg.ownership is not None
    assert cfg.ownership.owner_scope == "study_campaign"
    assert cfg.ownership.study_id == "stress_ethanol_cipro_growth"
    assert cfg.ownership.dataset_id == "usr_prom_eth_cip_opal_candidates"
    assert cfg.ownership.portable is False
    assert cfg.model.name == "random_forest"
    assert {view.id: view.objective.params["target_mask"] for view in cfg.selection_views} == expected
    assert all(view.selection.name == "top_n" for view in cfg.selection_views)
    assert all(view.selection.params["top_k"] == 6 for view in cfg.selection_views)
    assert all(view.selection.params["require_exact_top_k"] is True for view in cfg.selection_views)
    assert all(view.objective.name == "multistate_response_behavior_v1" for view in cfg.selection_views)
    assert all(view.selection.params["score_ref"] == "behavior_score" for view in cfg.selection_views)
    assert cfg.selection_batch.deduplicate_by == "sequence"
    assert cfg.selection_batch.expected_unique_count == 18


def test_study_docs_use_candidate_feature_table_name() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            STUDY_DOCS / "record" / "campaign.yaml",
            STUDY_DOCS / "record" / "datasets.yaml",
            STUDY_DOCS / "operations" / "ops.study.yaml",
            STUDY_DOCS / "operations" / "contract" / "surfaces" / "artifacts.yaml",
            STUDY_DOCS / "routes" / "README.md",
            STUDY_DOCS / "routes" / "decision" / "opal" / "README.md",
            STUDY_DOCS / "contexts" / "opal" / "candidate-table.md",
            STUDY_DOCS / "record" / "status.md",
            BATCH0_README,
        ]
    )

    assert "usr_prom_eth_cip_matrix" not in docs
    assert "usr_prom_eth_cip_opal_candidates" in docs
    assert "opal_candidate_feature_table" in docs
    assert "one shared USR `opal_candidate_feature_table`" in docs
    assert "Do not mint one USR dataset per campaign" in docs
    assert "one campaign fits the shared eight-output phenotype model" in docs
    assert "named selection views" in docs
    assert "records-path lock" in docs
    assert "data.location.kind: usr" in docs
    assert "observed assay labels as study-level truth" in docs
    assert "three digest-pinned SFXI source runs remain" in docs
    assert "response-window label snapshot" in docs
    assert "digest-pinned study-provenance manifest" in docs
    assert "_opal/observed_labels.parquet" in docs
    assert "_opal/response_window_labels_v5/observed_labels.parquet" in docs
    assert "decision.opal.batch0.provenance" in docs
    assert "raw Infer vector concat" in docs


def test_opal_candidate_table_contract_tracks_round0_augmented_materialization() -> None:
    artifacts = yaml.safe_load(
        (STUDY_DOCS / "operations" / "contract" / "surfaces" / "artifacts.yaml").read_text(encoding="utf-8")
    )
    readiness = yaml.safe_load(
        (
            STUDY_DOCS / "operations" / "contract" / "readiness" / "checks" / "opal_candidate_table_pre_assay.yaml"
        ).read_text(encoding="utf-8")
    )

    artifact = artifacts["opal_candidate_feature_table"]
    check = readiness["checks"]["opal_candidate_table_pre_assay"][0]

    assert artifact["row_count"] == 157185
    assert artifact["composition"] == {
        "generated_promoter_candidates": 157160,
        "measured_pdual10_sfxi_reference_rows": 23,
        "measured_pdual10_control_rows": 2,
    }
    assert artifact["source_population"] == "dense_generated_promoters_plus_measured_reader_round0_rows"
    assert artifact["sfxi_source_label_pool_state"] == "present_35_rows"
    assert artifact["response_window_label_sidecar_state"] == "verified_27_labels"
    promotion = artifacts[artifact["response_window_label_promotion_artifact"]]
    assert promotion["schema_version"] == "opal.observed_label_promotion.v1"
    assert promotion["y_space"] == "reader_response_window_vector_v1"
    assert promotion["reduction_id"] == "event_logmean_4_8h_post"
    assert promotion["label_rows"] == 27
    assert "archive_sfxi_reference_control_rows" not in artifact.get("excludes", "")
    assert check["target_rows"] == artifact["row_count"]
    assert check["row_count_mode"] == "exact"
    assert check["check_group"] == "opal_candidate_table"
    assert "measured Reader round-0 rows" in check["summary"]


def test_opal_round0_ops_phase_routes_to_readonly_candidate_review() -> None:
    operations = STUDY_DOCS / "operations"
    lifecycle = yaml.safe_load((operations / "contract" / "lifecycle" / "mode.yaml").read_text(encoding="utf-8"))
    phases = yaml.safe_load((operations / "contract" / "lifecycle" / "phases.yaml").read_text(encoding="utf-8"))
    bindings = yaml.safe_load(
        (operations / "contract" / "readiness" / "group-bindings.yaml").read_text(encoding="utf-8")
    )
    readiness = yaml.safe_load(
        (operations / "contract" / "readiness" / "checks" / "opal_round0_candidate_review.yaml").read_text(
            encoding="utf-8"
        )
    )
    surfaces = yaml.safe_load(
        (operations / "contract" / "surfaces" / "execution" / "commands" / "opal" / "round0-review.yaml").read_text(
            encoding="utf-8"
        )
    )
    artifacts = yaml.safe_load((operations / "contract" / "surfaces" / "artifacts.yaml").read_text(encoding="utf-8"))

    phase_by_id = {phase["id"]: phase for phase in phases}
    assert lifecycle["current_phase"] == {"strategy": "explicit", "id": "opal_round0_candidate_review"}
    assert phase_by_id["opal_candidate_table_pre_assay"]["status"] == "complete"
    review_phase = phase_by_id["opal_round0_candidate_review"]
    assert review_phase["status"] == "in_progress"
    assert review_phase["next_surface"].endswith("notebooks/opal_secg_msrb_greedy_analysis.py")
    assert "does not authorize synthesis" in review_phase["notes"]
    assert bindings["group_phase_bindings"]["opal"] == "opal_round0_candidate_review"

    checks = readiness["checks"]["opal_round0_candidate_review"]
    assert {check["check_id"] for check in checks} == {
        "opal.response_window_labels.promotion",
        "opal.round0.run_context",
        "opal.round0.selection_batch",
        "opal.campaign.validate",
        "opal.round0.selection_batch.load",
        "opal.round0.ethanol.verify_outputs",
        "opal.round0.ciprofloxacin.verify_outputs",
        "opal.round0.and.verify_outputs",
    }
    command_surface_ids = {check["surface"] for check in checks if check["kind"] == "command"}
    for surface_id in command_surface_ids:
        assert surfaces[surface_id]["writes_artifacts"] is False

    assert surfaces["opal_campaign_validate_json"]["argv"][-1] == "--json"
    for surface_id in {
        "opal_selection_batch_round0_json",
        "opal_verify_outputs_round0_ethanol_json",
        "opal_verify_outputs_round0_ciprofloxacin_json",
        "opal_verify_outputs_round0_and_json",
    }:
        argv = surfaces[surface_id]["argv"]
        assert argv[argv.index("--round") + 1] == "0"

    promotion = artifacts["opal_response_window_label_promotion"]
    assert promotion["label_rows"] == 27
    assert promotion["ref"].startswith("repo:")
    selection_batch = artifacts["opal_round0_selection_batch"]
    assert selection_batch["row_count"] == 18
    assert selection_batch["deduplicate_by"] == "sequence"
    assert selection_batch["authorization_state"] == "candidate_review_only"
    assert selection_batch["ref"].startswith("repo:")


def test_study_routes_expose_opal_notebook_generate_as_campaign_viewer() -> None:
    routes = (STUDY_DOCS / "routes" / "README.md").read_text(encoding="utf-8")
    opal_route = (STUDY_DOCS / "routes" / "decision" / "opal" / "README.md").read_text(encoding="utf-8")
    opal_commands = (STUDY_DOCS / "routes" / "decision" / "opal" / "campaign-commands.md").read_text(encoding="utf-8")
    pipeline = yaml.safe_load(
        (STUDY_DOCS / "operations" / "runtime" / "command-groups" / "pipeline.yaml").read_text(encoding="utf-8")
    )
    opal_pipeline = pipeline["study_pipeline"]["opal"]
    opal_notebook = opal_pipeline["notebook"]

    assert "routes/decision/opal/README.md" in routes
    assert "Campaign configs and commands" in opal_route
    assert "Read-only campaign verification" in opal_commands
    assert "Notebook review" in opal_commands
    assert "Notebook generation writes the notebook artifact" in opal_commands
    assert "uv run opal notebook generate" in opal_commands
    assert "uv run opal notebook run" in opal_commands
    assert "uv run opal status" in opal_commands
    assert "unified notebook displays named selection views" in opal_route
    assert "opal selection-batch show" in opal_commands
    assert "decision.opal.batch0.provenance" in opal_route
    assert "studies.stress-ethanol-cipro-growth.status" in routes
    assert opal_notebook["role"] == "campaign_specific_artifact_viewer"
    assert opal_notebook["pre_run_execution_safe"] is True
    assert opal_notebook["mutates_notebook"] is True
    assert "review_json_command" not in opal_pipeline
    assert "uv run opal review" in opal_pipeline["review_materialize_json_command"]
    assert opal_pipeline["review_materialization_writes_artifacts"] is True
    assert opal_pipeline["review_materialization_output_root"].endswith("outputs/review")


def test_study_pipeline_exposes_only_readonly_synthesis_handoff_previews() -> None:
    operations = STUDY_DOCS / "operations"
    pipeline = yaml.safe_load((operations / "runtime" / "command-groups" / "pipeline.yaml").read_text(encoding="utf-8"))
    command_catalog = yaml.safe_load(
        (
            operations / "contract" / "surfaces" / "execution" / "commands" / "opal" / "synthesis-handoffs.yaml"
        ).read_text(encoding="utf-8")
    )
    opal_pipeline = pipeline["study_pipeline"]["opal"]

    assert opal_pipeline["synthesis_authorization"] == "not_granted"
    handoff_commands = {
        key: value
        for key, value in opal_pipeline.items()
        if key.startswith("synthesis_handoff_") and key.endswith("_command")
    }
    assert set(handoff_commands) == {
        "synthesis_handoff_round_draft_preview_command",
        "synthesis_handoff_round_preview_command",
        "synthesis_handoff_source_evidence_preview_command",
    }
    assert all("--write" not in command for command in handoff_commands.values())
    assert command_catalog
    assert all(surface["writes_artifacts"] is False for surface in command_catalog.values())
    assert all("--write" not in surface["argv"] for surface in command_catalog.values())


def test_study_status_catalog_handoff_routes_to_opal_without_generic_feature_matrix() -> None:
    registry = yaml.safe_load(
        (STUDY_DOCS / "operations" / "catalog" / "contracts" / "registry" / "status.registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    relation_targets = {relation["target"] for relation in registry["relations"]}

    assert "opal.downstream.usr-infer-x-active-learning" in relation_targets
    assert "usr.data-plane.promoter-feature-matrix" not in relation_targets


def test_study_route_map_uses_progressive_disclosure_for_opal_and_latentdna() -> None:
    routes_path = STUDY_DOCS / "routes" / "README.md"
    routes = routes_path.read_text(encoding="utf-8")
    opal_route = (STUDY_DOCS / "routes" / "decision" / "opal" / "README.md").read_text(encoding="utf-8")
    opal_context = (STUDY_DOCS / "contexts" / "opal" / "candidate-table.md").read_text(encoding="utf-8")
    latentdna_route = (STUDY_DOCS / "routes" / "analysis" / "latentdna.md").read_text(encoding="utf-8")

    assert len(routes.splitlines()) <= 140
    assert "routes/decision/opal/README.md" in routes
    assert "routes/analysis/latentdna.md" in routes
    assert "opal_round0_candidate_review" in opal_route
    assert "response-window label snapshot" in opal_context
    assert "intermediate_embedding_7b_context_anchor_mean_bidir_concat" in latentdna_route
