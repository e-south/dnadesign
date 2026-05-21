"""
Focused contracts for the stress/ethanol/ciprofloxacin OPAL batch-0 handoff.
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
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.candidate_table import (
    main as candidate_table_main,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.candidate_table import (
    materialize_configured_candidate_feature_table,
    validate_candidate_feature_table,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance import (
    audit_candidate_lineage,
    show_candidate_lineage,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.select import (
    REQUIRED_REVIEW_COLUMNS,
    load_sampling_config,
    select_batch0,
    validate_configured_candidate_feature_table,
    validate_selected_ids_against_candidate_feature_table,
)
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.select import (
    main as batch0_select_main,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLING = REPO_ROOT / "src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml"
CAMPAIGN_ROOT = REPO_ROOT / "src/dnadesign/opal/campaigns"
STUDY_DOCS = REPO_ROOT / "docs/studies/stress_ethanol_cipro_growth"
BATCH0_README = REPO_ROOT / "src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/README.md"


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
    }


def _row(
    row_id: str,
    *,
    plan: str,
    regulators: str,
    sigma: str = "f",
    spacer: int = 16,
    ethanol: float = 0.2,
    cipro: float = 0.2,
    dual: float = 0.2,
    tier: int = 1,
) -> dict[str, object]:
    return {
        "id": row_id,
        "sequence": "ACGT" * 15,
        "canonical_densegen_plan": plan,
        "regulator_composition": regulators,
        "sigma35_variant": sigma,
        "spacer_length": spacer,
        "target_margin": max(ethanol, cipro, dual),
        "synthetic_margin_ethanol_vs_background": ethanol,
        "synthetic_margin_cipro_vs_background": cipro,
        "synthetic_margin_dual_vs_background": dual,
        "sig35_margin_f_vs_b": 0.4,
        "tfbs_summary": f"{regulators}@{spacer}",
        "motif_score_summary": f"tier={tier}",
        "tfbs_offset_summary": str(spacer),
        "tfbs_orientation_summary": "fwd",
        "motif_tier_summary": str(tier),
        "x_provenance": "intermediate_embedding_7b_context_anchor_mean_bidir_concat",
    }


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
    rows = [
        _row("eth_baer_f1", plan="ethanol", regulators="baeR", sigma="f", ethanol=0.91),
        _row("eth_cpxr_e1", plan="ethanol", regulators="cpxR", sigma="e", ethanol=0.90),
        _row("eth_baer_e2", plan="ethanol", regulators="baeR", sigma="e", ethanol=0.89),
        _row("eth_cpxr_f2", plan="ethanol", regulators="cpxR", sigma="f", ethanol=0.88),
        _row("eth_expl_d", plan="ethanol", regulators="baeR", sigma="d", ethanol=0.87),
        _row("eth_dual_cpxr", plan="ethanol_ciprofloxacin", regulators="cpxR+lexA", sigma="f", ethanol=0.86),
        _row("eth_dual_baer", plan="ethanol_ciprofloxacin", regulators="baeR+lexA", sigma="e", ethanol=0.85),
        _row("cip_lexa_f1", plan="ciprofloxacin", regulators="lexA", sigma="f", cipro=0.91),
        _row("cip_lexa_e1", plan="ciprofloxacin", regulators="lexA", sigma="e", cipro=0.90),
        _row("cip_lexa_f2", plan="ciprofloxacin", regulators="lexA", sigma="f", cipro=0.89),
        _row("cip_lexa_e2", plan="ciprofloxacin", regulators="lexA", sigma="e", cipro=0.88),
        _row("cip_lexa_d", plan="ciprofloxacin", regulators="lexA", sigma="d", cipro=0.87),
        _row("cip_dual_cpxr", plan="ethanol_ciprofloxacin", regulators="cpxR+lexA", sigma="f", cipro=0.86),
        _row("cip_dual_baer", plan="ethanol_ciprofloxacin", regulators="baeR+lexA", sigma="e", cipro=0.85),
        _row("and_cpxr_1", plan="ethanol_ciprofloxacin", regulators="cpxR+lexA", sigma="f", dual=0.96),
        _row("and_cpxr_2", plan="ethanol_ciprofloxacin", regulators="cpxR+lexA", sigma="e", dual=0.95),
        _row("and_cpxr_3", plan="ethanol_ciprofloxacin", regulators="cpxR+lexA", sigma="d", dual=0.94),
        _row("and_baer_1", plan="ethanol_ciprofloxacin", regulators="baeR+lexA", sigma="f", dual=0.93),
        _row("and_baer_2", plan="ethanol_ciprofloxacin", regulators="baeR+lexA", sigma="e", dual=0.92),
        _row("and_baer_3", plan="ethanol_ciprofloxacin", regulators="baeR+lexA", sigma="c", dual=0.91),
        _row("negative_prior", plan="ethanol", regulators="baeR", sigma="f", ethanol=-0.1),
    ]

    selected = select_batch0(pd.DataFrame(rows), config)

    assert set(REQUIRED_REVIEW_COLUMNS).issubset(selected.columns)
    assert selected.groupby("campaign").size().to_dict() == {
        "stress_eth_cip_and_rf_sfxi_topn": 6,
        "stress_eth_cip_cipro_rf_sfxi_topn": 6,
        "stress_eth_cip_ethanol_rf_sfxi_topn": 6,
    }
    assert not selected["id"].duplicated().any()
    assert "negative_prior" not in set(selected["id"])

    ethanol = selected[selected["campaign"] == "stress_eth_cip_ethanol_rf_sfxi_topn"]
    assert ethanol["setpoint"].map(tuple).unique().tolist() == [(0, 1, 0, 1)]
    assert ethanol["canonical_densegen_plan"].value_counts().to_dict() == {
        "ethanol": 4,
        "ethanol_ciprofloxacin": 2,
    }

    cipro = selected[selected["campaign"] == "stress_eth_cip_cipro_rf_sfxi_topn"]
    assert cipro["setpoint"].map(tuple).unique().tolist() == [(0, 0, 1, 1)]
    assert cipro["canonical_densegen_plan"].value_counts().to_dict() == {
        "ciprofloxacin": 4,
        "ethanol_ciprofloxacin": 2,
    }
    assert cipro["regulator_composition"].str.contains("lexA").all()

    and_gate = selected[selected["campaign"] == "stress_eth_cip_and_rf_sfxi_topn"]
    assert and_gate["setpoint"].map(tuple).unique().tolist() == [(0, 0, 0, 1)]
    assert and_gate["canonical_densegen_plan"].unique().tolist() == ["ethanol_ciprofloxacin"]
    assert and_gate["regulator_composition"].value_counts().to_dict() == {
        "baeR+lexA": 3,
        "cpxR+lexA": 3,
    }
    assert and_gate["sigma35_variant"].isin(["c", "d"]).any()


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

    assert report == {"row_count": 2, "x_dim": 2}
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
    ) == {"row_count": 2, "x_dim": 2}


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


def test_candidate_feature_table_validation_rejects_blank_densegen_provenance(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    provenance = _candidate_provenance(1)
    provenance["densegen__sampling_library_hash"] = [""]
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

    with pytest.raises(ValueError, match="densegen__sampling_library_hash"):
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

    assert report == {"row_count": 1, "x_dim": 2}


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
        "stress_eth_cip_ethanol_rf_sfxi_topn": [0, 1, 0, 1],
        "stress_eth_cip_cipro_rf_sfxi_topn": [0, 0, 1, 1],
        "stress_eth_cip_and_rf_sfxi_topn": [0, 0, 0, 1],
    }
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"

    for slug, setpoint in expected.items():
        cfg = load_config(CAMPAIGN_ROOT / slug / "configs/campaign.yaml")
        assert cfg.campaign.slug == slug
        assert cfg.data.location.kind == "usr"
        assert cfg.data.location.dataset == "usr_prom_eth_cip_opal_candidates"
        assert cfg.data.location.path.endswith("src/dnadesign/usr/datasets")
        assert cfg.data.x_column_name == x_col
        assert cfg.data.y_column_name == f"opal__{slug}__y"
        assert cfg.data.y_expected_length == 8
        assert cfg.labels.source.kind == "usr_sidecar"
        assert cfg.labels.source.dataset == "usr_prom_eth_cip_opal_candidates"
        assert cfg.labels.source.path == "_opal/observed_labels.parquet"
        assert cfg.labels.y_space == "sfxi_vec8"
        assert cfg.labels.round_column == "observed_round"
        assert cfg.labels.dedup_policy == "latest_by_round"
        assert cfg.writeback.prediction_records == "ledger_only"
        assert cfg.ownership is not None
        assert cfg.ownership.owner_scope == "study_fixture"
        assert cfg.ownership.study_id == "stress_ethanol_cipro_growth"
        assert cfg.ownership.dataset_id == "usr_prom_eth_cip_opal_candidates"
        assert cfg.ownership.portable is False
        assert cfg.model.name == "random_forest"
        assert cfg.selection.selection.name == "top_n"
        assert cfg.selection.selection.params["top_k"] == 6
        assert cfg.selection.selection.params["score_ref"] == "sfxi_v1/sfxi"
        assert cfg.objectives.objectives[0].params["setpoint_vector"] == setpoint


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
    assert "OPAL-writeback surface" in docs
    assert "records-path lock" in docs
    assert "data.location.kind: local" in docs
    assert "observed SFXI labels as study-level assay truth" in docs
    assert "Legacy OPAL training labels are campaign-slug scoped" in docs
    assert "observed-label store" in docs
    assert "_opal/observed_labels.parquet" in docs
    assert "opal_batch0.provenance" in docs
    assert "raw Infer vector concat" in docs


def test_study_routes_expose_opal_notebook_generate_as_campaign_viewer() -> None:
    routes = (STUDY_DOCS / "routes" / "README.md").read_text(encoding="utf-8")
    opal_route = (STUDY_DOCS / "routes" / "decision" / "opal" / "README.md").read_text(encoding="utf-8")
    opal_commands = (STUDY_DOCS / "routes" / "decision" / "opal" / "campaign-commands.md").read_text(encoding="utf-8")
    pipeline = yaml.safe_load(
        (STUDY_DOCS / "operations" / "runtime" / "command-groups" / "pipeline.yaml").read_text(encoding="utf-8")
    )
    opal_notebook = pipeline["study_pipeline"]["opal"]["notebook"]

    assert "routes/decision/opal/README.md" in routes
    assert "Campaign configs and commands" in opal_route
    assert "Pre-run campaign viewer generation" in opal_commands
    assert "uv run opal notebook generate" in opal_commands
    assert "uv run opal notebook run" in opal_commands
    assert "Post-run status command" in opal_commands
    assert "campaign-specific artifact viewer" in opal_route
    assert "Per-ID provenance trace" in opal_commands
    assert "opal_batch0.provenance" in opal_route
    assert "studies.stress-ethanol-cipro-growth.status" in routes
    assert opal_notebook["role"] == "campaign_specific_artifact_viewer"
    assert opal_notebook["pre_run_execution_safe"] is True
    assert opal_notebook["mutates_notebook"] is True


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
    assert "candidate_table_materialized_pre_assay" in opal_route
    assert "observed-label store" in opal_context
    assert "intermediate_embedding_7b_context_anchor_mean_bidir_concat" in latentdna_route
