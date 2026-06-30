"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_config_objectives_v2.py

Validates v2 config parsing for multi-objective score channel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.core.utils import ConfigError


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_config_accepts_objectives_list_and_score_ref(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
  metadata:
    scenario_kind: positive
    split_id: random
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
  - { name: sfxi_v1, params: { setpoint_vector: [0, 0, 0, 1], scaling: { min_n: 1 } } }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)
    assert len(cfg.objectives.objectives) == 2
    assert cfg.selection.selection.params["score_ref"] == "scalar_identity_v1/scalar"
    assert cfg.campaign.metadata == {"scenario_kind": "positive", "split_id": "random"}


def test_load_config_accepts_spop_objective_channel(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "reader_spop_endpoint_dose_mean_v1"
  y_expected_length: 1
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: spop_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "spop_v1/spop", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)

    assert cfg.objectives.objectives[0].name == "spop_v1"
    assert cfg.selection.selection.params["score_ref"] == "spop_v1/spop"


def test_load_config_accepts_candidate_scope_id_list(tmp_path: Path) -> None:
    scope_path = tmp_path / "scope.parquet"
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        f"""
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: {{ kind: local, path: "./records.parquet" }}
  x_column_name: "X"
  y_column_name: "Y"
  candidate_scope:
    kind: id_list
    path: "{scope_path}"
    id_column: id
transforms_x: {{ name: identity, params: {{}} }}
transforms_y: {{ name: scalar_from_table_v1, params: {{}} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params:
    top_k: 2
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
""".strip(),
    )

    cfg = load_config(cfg_path)

    assert cfg.data.candidate_scope is not None
    assert cfg.data.candidate_scope.kind == "id_list"
    assert cfg.data.candidate_scope.path == str(scope_path.resolve())
    assert cfg.data.candidate_scope.id_column == "id"


def test_load_config_resolves_candidate_scope_from_campaign_root_for_configs_dir(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    configs_dir = campaign_root / "configs"
    scope_path = campaign_root / "scopes" / "heldout_ids.csv"
    scope_path.parent.mkdir(parents=True)
    configs_dir.mkdir(parents=True)
    cfg_path = _write_config(
        configs_dir / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
  candidate_scope:
    kind: id_list
    path: "scopes/heldout_ids.csv"
    id_column: id
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params:
    top_k: 2
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
""".strip(),
    )

    cfg = load_config(cfg_path)

    assert cfg.data.candidate_scope is not None
    assert cfg.data.candidate_scope.path == str(scope_path.resolve())


def test_load_config_accepts_candidate_eligibility_restriction_site_rule(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
candidate_eligibility:
  rules:
    - name: restriction_site_exclusion
      params:
        sequence_column: sequence
        scan_space: final_assembled_insert
        assembly_strategy_ref: sfxi_promoter_insert:v1
        left_flank: accgggatcctgcag
        right_flank: tgagggaattcgcga
        expected_core_length: 60
        min_remaining_candidates: 1
        forbidden_sites:
          - enzyme: BamHI
            motif: GGATCC
            allowed_regions: [left_flank]
          - enzyme: EcoRI
            motif: GAATTC
            allowed_regions: [right_flank]
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params:
    top_k: 2
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
""".strip(),
    )

    cfg = load_config(cfg_path)

    assert len(cfg.candidate_eligibility.rules) == 1
    rule = cfg.candidate_eligibility.rules[0]
    assert rule.name == "restriction_site_exclusion"
    assert rule.params["assembly_strategy_ref"] == "sfxi_promoter_insert:v1"
    assert rule.params["forbidden_sites"][0]["enzyme"] == "BamHI"


def test_load_config_accepts_sfxi_uncertainty_method(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: sfxi_vec8_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: sfxi_v1, params: { setpoint_vector: [0, 0, 0, 1], uncertainty_method: analytical, scaling: { min_n: 1 } } }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "sfxi_v1/sfxi", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)
    assert cfg.objectives.objectives[0].params["uncertainty_method"] == "analytical"


def test_load_config_accepts_usr_sidecar_label_source(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: usr, path: "./usr/datasets", dataset: demo_candidates }
  x_column_name: "X"
  y_column_name: "opal__demo__y"
labels:
  source:
    kind: usr_sidecar
    dataset: demo_candidates
    path: _opal/observed_labels.parquet
  y_space: sfxi_vec8
  id_column: id
  round_column: observed_round
  batch_column: batch_id
  dedup_policy: latest_by_round
writeback:
  prediction_records: ledger_only
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)
    assert cfg.labels.source.kind == "usr_sidecar"
    assert cfg.labels.source.dataset == "demo_candidates"
    assert cfg.labels.y_space == "sfxi_vec8"
    assert cfg.writeback.prediction_records == "ledger_only"


def test_load_config_accepts_artifact_retention_block(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
artifact_retention:
  mode: production_review
  prediction_ledger: latest_full_plus_selected_history
  plot_tidy_data: compact
  model_artifacts: latest
  tabular_format: parquet_zstd
  max_estimated_bytes: 50000000000
  fail_if_estimate_exceeds: true
  final_round: 11
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)

    assert cfg.artifact_retention.mode == "production_review"
    assert cfg.artifact_retention.prediction_ledger == "latest_full_plus_selected_history"
    assert cfg.artifact_retention.plot_tidy_data == "compact"
    assert cfg.artifact_retention.model_artifacts == "latest"
    assert cfg.artifact_retention.tabular_format == "parquet_zstd"
    assert cfg.artifact_retention.final_round == 11


def test_load_config_rejects_usr_sidecar_without_explicit_prediction_writeback(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: usr, path: "./usr/datasets", dataset: demo_candidates }
  x_column_name: "X"
  y_column_name: "opal__demo__y"
labels:
  source:
    kind: usr_sidecar
    dataset: demo_candidates
    path: _opal/observed_labels.parquet
  y_space: sfxi_vec8
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError, match="writeback.prediction_records"):
        _ = load_config(cfg_path)


def test_load_config_rejects_usr_sidecar_label_source_for_different_dataset(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: usr, path: "./usr/datasets", dataset: demo_candidates }
  x_column_name: "X"
  y_column_name: "opal__demo__y"
labels:
  source:
    kind: usr_sidecar
    dataset: other_candidates
    path: _opal/observed_labels.parquet
  y_space: sfxi_vec8
writeback:
  prediction_records: ledger_only
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError, match="same dataset"):
        _ = load_config(cfg_path)


def test_load_config_rejects_sfxi_analytical_with_non_unit_exponents(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: sfxi_vec8_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - name: sfxi_v1
    params:
      setpoint_vector: [0, 0, 0, 1]
      logic_exponent_beta: 1.1
      intensity_exponent_gamma: 1.0
      uncertainty_method: analytical
      scaling: { min_n: 1 }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "sfxi_v1/sfxi", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError, match="uncertainty_method=analytical requires"):
        _ = load_config(cfg_path)


def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
objectives:
  - { name: scalar_identity_v1, params: {} }
""".strip(),
    )

    with pytest.raises(ConfigError, match="Duplicate key in YAML"):
        _ = load_config(cfg_path)


@pytest.mark.parametrize("invalid_method", ["bogus", "auto"])
def test_load_config_rejects_invalid_sfxi_uncertainty_method(tmp_path: Path, invalid_method: str) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: sfxi_vec8_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - name: sfxi_v1
    params:
      setpoint_vector: [0, 0, 0, 1]
      uncertainty_method: INVALID_METHOD
      scaling: { min_n: 1 }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "sfxi_v1/sfxi", objective_mode: maximize, tie_handling: competition_rank }
""".replace("INVALID_METHOD", invalid_method).strip(),
    )

    with pytest.raises(ConfigError, match="uncertainty_method"):
        _ = load_config(cfg_path)


def test_load_config_rejects_missing_selection_score_ref(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError, match="score_ref"):
        _ = load_config(cfg_path)


def test_load_config_rejects_unknown_selection_param_key(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: random_forest, params: { n_estimators: 5, random_state: 0 } }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params:
    top_k: 2
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
    unknown_key: maximize
""".strip(),
    )

    with pytest.raises(ConfigError, match="unknown_key"):
        _ = load_config(cfg_path)


def test_load_config_accepts_gaussian_process_kernel_block(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model:
  name: gaussian_process
  params:
    alpha: 1.0e-6
    normalize_y: true
    n_restarts_optimizer: 2
    kernel:
      name: matern
      length_scale: 0.5
      nu: 1.5
      with_white_noise: true
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    cfg = load_config(cfg_path)
    assert cfg.model.name == "gaussian_process"
    assert cfg.model.params["kernel"]["name"] == "matern"


def test_load_config_rejects_unknown_gaussian_process_kernel_name(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model:
  name: gaussian_process
  params:
    kernel:
      name: bad_kernel
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError, match="kernel"):
        _ = load_config(cfg_path)


def test_load_config_rejects_unknown_model_plugin_name(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "campaign.yaml",
        """
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: { kind: local, path: "./records.parquet" }
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }
model: { name: unknown_model_v99, params: {} }
objectives:
  - { name: scalar_identity_v1, params: {} }
selection:
  name: top_n
  params: { top_k: 2, score_ref: "scalar_identity_v1/scalar", objective_mode: maximize, tie_handling: competition_rank }
""".strip(),
    )

    with pytest.raises(ConfigError) as exc:
        _ = load_config(cfg_path)
    msg = str(exc.value)
    assert "Unknown model plugin" in msg
    assert "Available plugins:" in msg
    assert "gaussian_process" in msg
    assert "random_forest" in msg
