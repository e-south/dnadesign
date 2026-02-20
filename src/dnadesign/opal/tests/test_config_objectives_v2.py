"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_config_objectives_v2.py

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
