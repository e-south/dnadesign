"""
Focused contracts for the stress/ethanol/ciprofloxacin OPAL batch-0 handoff.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.opal import load_config
from dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.select import (
    REQUIRED_REVIEW_COLUMNS,
    load_sampling_config,
    select_batch0,
    validate_candidate_feature_table,
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

    pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
            "densegen__plan": ["ethanol__sig35=f", "ciprofloxacin__sig35=e"],
        }
    ).to_parquet(records)
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
        }
    ).to_parquet(records)

    with pytest.raises(ValueError, match="fixed-length"):
        validate_candidate_feature_table(records_path=records, x_column=x_col)


def test_configured_candidate_feature_table_validation_resolves_repo_paths(tmp_path: Path) -> None:
    records = tmp_path / "usr" / "datasets" / "demo" / "records.parquet"
    view_rows = tmp_path / "latentdna" / "views" / "rows.parquet"
    records.parent.mkdir(parents=True)
    view_rows.parent.mkdir(parents=True)
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
        }
    ).to_parquet(records)
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
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "bio_type": ["dna", "dna"],
            "sequence": ["AAAA", "CCCC"],
            "alphabet": ["dna_4", "dna_4"],
            x_col: [[0.1, 0.2], [0.3, 0.4]],
        }
    ).to_parquet(records)
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


def test_selected_ids_must_exist_in_candidate_feature_table(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    x_col = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
    pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAAA"],
            "alphabet": ["dna_4"],
            x_col: [[0.1, 0.2]],
        }
    ).to_parquet(records)

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
        assert cfg.model.name == "random_forest"
        assert cfg.selection.selection.name == "top_n"
        assert cfg.selection.selection.params["top_k"] == 6
        assert cfg.selection.selection.params["score_ref"] == "sfxi_v1/sfxi"
        assert cfg.objectives.objectives[0].params["setpoint_vector"] == setpoint


def test_study_docs_use_candidate_feature_table_name() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            STUDY_DOCS / "campaign.yaml",
            STUDY_DOCS / "datasets.yaml",
            STUDY_DOCS / "ops.study.yaml",
            STUDY_DOCS / "routes.md",
            STUDY_DOCS / "status.md",
        ]
    )

    assert "usr_prom_eth_cip_matrix" not in docs
    assert "usr_prom_eth_cip_opal_candidates" in docs
    assert "opal_candidate_feature_table" in docs


def test_study_routes_expose_opal_notebook_generate_as_campaign_viewer() -> None:
    routes = (STUDY_DOCS / "routes.md").read_text(encoding="utf-8")
    pipeline = yaml.safe_load((STUDY_DOCS / "pipeline.yaml").read_text(encoding="utf-8"))
    opal_notebook = pipeline["study_pipeline"]["opal"]["notebook"]

    assert "Pre-run-safe campaign viewer" in routes
    assert "uv run opal notebook generate" in routes
    assert "uv run opal notebook run" in routes
    assert "Post-run status command" in routes
    assert "campaign-specific artifact viewer" in routes
    assert "studies.stress-ethanol-cipro-growth.status" in routes
    assert opal_notebook["role"] == "campaign_specific_artifact_viewer"
    assert opal_notebook["pre_run_safe"] is True
