"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tf_count_active_targets.py

Regression tests for tf count active targets studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    ORACLE_ID,
    Path,
    ProbeArtifactLayout,
    RunSpec,
    _make_training_input_for_run,
    _write_campaign_config,
    pd,
    yaml,
)


def test_make_training_input_for_tf_count_materializes_compact_objectives(tmp_path: Path) -> None:
    run = RunSpec(
        campaign_key="ethanol",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="tf_count_ethanol_positive_random_id",
        target_class="tf_count__cpxR_plus_baeR",
        workdir=tmp_path / "workdir",
        config_path=tmp_path / "workdir" / "configs" / "campaign.yaml",
        label_input_path=tmp_path / "workdir" / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=tmp_path / "workdir" / "observed_labels.parquet",
        label_family_id="tf_family_count",
        target_channel="tf_count__cpxR_plus_baeR",
    )
    labels = pd.DataFrame(
        {
            "id": ["id-1"],
            "sequence": ["AAAA"],
            "tf_family__lexA__count": [2],
            "tf_family__cpxR__count": [3],
            "tf_family__baeR__count": [5],
        }
    )

    training = _make_training_input_for_run(labels, ["id-1"], run)

    assert training.to_dict(orient="records") == [
        {
            "id": "id-1",
            "sequence": "AAAA",
            "tf_count__lexA": 2.0,
            "tf_count__cpxR_plus_baeR": 8.0,
            "tf_count__lexA_plus_cpxR_plus_baeR": 10.0,
        }
    ]


def test_write_campaign_config_uses_generic_vector_contract_for_tf_count(tmp_path: Path) -> None:
    source_config = tmp_path / "src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml"
    source_config.parent.mkdir(parents=True)
    source_config.write_text(yaml.safe_dump(_minimal_source_config(), sort_keys=False), encoding="utf-8")
    run_root = tmp_path / "probe"
    layout = ProbeArtifactLayout(run_root)
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="tf_count_cipro_positive_random_id",
        target_class="tf_count__lexA",
        workdir=layout.campaign_workdir("tf_count_cipro_positive_random_id"),
        config_path=layout.campaign_config_path("tf_count_cipro_positive_random_id"),
        label_input_path=layout.campaign_label_input_path("tf_count_cipro_positive_random_id"),
        sidecar_path=layout.campaign_sidecar_path("tf_count_cipro_positive_random_id", "random_id"),
        label_family_id="tf_family_count",
        target_channel="tf_count__lexA",
    )

    _write_campaign_config(tmp_path, run, run_root)

    cfg = yaml.safe_load(run.config_path.read_text(encoding="utf-8"))
    plots = yaml.safe_load((run.config_path.parent / "plots.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["candidate_scope"] == {
        "kind": "id_list",
        "path": str(layout.split_candidate_scope_path("random_id").resolve()),
        "id_column": "id",
    }
    assert cfg["data"]["y_expected_length"] == 3
    assert cfg["labels"]["y_space"] == "numeric_vector"
    assert cfg["transforms_y"]["name"] == "vector_from_table_v1"
    view = cfg["selection_views"][0]
    assert view["id"] == "primary"
    assert view["objective"]["name"] == "vector_channel_v1"
    assert view["objective"]["params"]["channel_name"] == "tf_count__lexA"
    assert view["selection"]["params"]["score_ref"] == "tf_count__lexA"
    assert "y_ops" not in cfg["training"]
    plot_names = {row["name"] for row in plots["plots"]}
    assert "selected_target_vector_summary" in plot_names
    assert all(not str(name).startswith("sfxi_") for name in plot_names)


def _minimal_source_config() -> dict[str, object]:
    return {
        "schema_version": "opal.campaign.v3",
        "campaign": {"name": "source", "slug": "source", "workdir": ".", "metadata": {}},
        "data": {"location": {"kind": "usr", "path": ".", "dataset": "src"}, "y_column_name": "y"},
        "labels": {"source": {"kind": "usr_sidecar", "dataset": "src", "path": "labels.parquet"}},
        "training": {"policy": {}, "y_ops": [{"name": "intensity_median_iqr", "params": {}}]},
        "transforms_x": {"name": "identity", "params": {}},
        "transforms_y": {"name": "vector_from_table_v1", "params": {}},
        "model": {"name": "random_forest", "params": {}},
        "selection_views": [
            {
                "id": "primary",
                "objective": {"name": "vector_channel_v1", "params": {}},
                "selection": {"name": "top_n", "params": {"top_k": 6, "score_ref": "channel"}},
            }
        ],
        "selection_batch": {"deduplicate_by": "id"},
        "scoring": {"score_batch_size": 1000},
        "safety": {},
    }
