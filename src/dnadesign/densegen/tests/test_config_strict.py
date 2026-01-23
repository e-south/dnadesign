from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from dnadesign.densegen.src.config import ConfigError, load_config

MIN_CONFIG = {
    "densegen": {
        "schema_version": "2.5",
        "run": {"id": "demo", "root": "."},
        "inputs": [
            {
                "name": "demo",
                "type": "binding_sites",
                "path": "inputs.csv",
            }
        ],
        "output": {
            "targets": ["parquet"],
            "schema": {"bio_type": "dna", "alphabet": "dna_4"},
            "parquet": {
                "path": "outputs/tables/dense_arrays.parquet",
                "deduplicate": True,
                "chunk_size": 128,
            },
        },
        "generation": {
            "sequence_length": 10,
            "quota": 1,
            "plan": [{"name": "default", "quota": 1}],
        },
        "solver": {"backend": "CBC", "strategy": "iterate", "options": []},
        "logging": {"log_dir": "outputs/logs"},
    }
}


def _write(cfg: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_requires_densegen_root(tmp_path: Path) -> None:
    cfg_path = _write({"foo": "bar"}, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_schema_version_required(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"].pop("schema_version", None)
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_schema_version_supported(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["schema_version"] = "9.9"
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_gap_fill_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["postprocess"] = {"gap_fill": {"mode": "off"}}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="gap_fill"):
        load_config(cfg_path)


def test_pad_config_accepts(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["postprocess"] = {
        "pad": {
            "mode": "off",
            "end": "5prime",
            "gc": {
                "mode": "range",
                "min": 0.4,
                "max": 0.6,
                "target": 0.5,
                "tolerance": 0.1,
                "min_pad_length": 4,
            },
            "max_tries": 2000,
        }
    }
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    load_config(cfg_path)


def test_plan_mixing_quota_and_fraction(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["plan"] = [
        {"name": "a", "quota": 1},
        {"name": "b", "fraction": 0.5},
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_usr_output_requires_root(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"] = {
        "targets": ["usr"],
        "usr": {"dataset": "demo", "chunk_size": 10, "allow_overwrite": False},
    }
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_usr_sequences_requires_root(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["inputs"] = [{"name": "demo", "type": "usr_sequences", "dataset": "demo"}]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_output_paths_must_live_under_outputs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"]["parquet"]["path"] = "dense_arrays.parquet"
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="output.parquet.path must be within outputs"):
        load_config(cfg_path)


def test_usr_root_must_live_under_outputs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"] = {
        "targets": ["usr"],
        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
        "usr": {"dataset": "demo", "root": "usr"},
    }
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="output.usr.root must be within outputs"):
        load_config(cfg_path)


def test_logging_dir_must_live_under_outputs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["logging"]["log_dir"] = "logs"
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="logging.log_dir must be within outputs"):
        load_config(cfg_path)


def test_plots_dir_must_live_under_outputs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["plots"] = {"out_dir": "plots"}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="plots.out_dir must be within outputs"):
        load_config(cfg_path)


def test_library_artifact_path_must_live_under_outputs(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["sampling"] = {
        "library_source": "artifact",
        "library_artifact_path": "libraries",
    }
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="sampling.library_artifact_path must be within outputs"):
        load_config(cfg_path)


def test_output_kind_is_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"] = {"kind": "parquet", "parquet": {"path": "outputs/tables/demo_parquet.parquet"}}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_solver_options_unknown_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["solver"]["options"] = ["UnknownOpt=1"]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="Unknown solver.options"):
        load_config(cfg_path)


def test_solver_options_allow_unknown(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["solver"]["options"] = ["UnknownOpt=1"]
    cfg["densegen"]["solver"]["allow_unknown_options"] = True
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    load_config(cfg_path)


def test_promoter_constraint_motif_validation(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["plan"] = [
        {
            "name": "bad",
            "quota": 1,
            "fixed_elements": {
                "promoter_constraints": [{"upstream": "TTGAZ", "downstream": "TATAAT", "spacer_length": [16, 18]}]
            },
        }
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_fimo_rejects_max_candidates(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["inputs"] = [
        {
            "name": "motifs",
            "type": "pwm_meme",
            "path": "inputs.meme",
            "sampling": {
                "strategy": "stochastic",
                "n_sites": 2,
                "oversample_factor": 2,
                "scoring_backend": "fimo",
                "pvalue_threshold": 1e-4,
                "max_candidates": 100,
                "mining": {"batch_size": 10},
            },
        }
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError, match="max_candidates is not used"):
        load_config(cfg_path)


def test_promoter_constraint_range_non_negative(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["plan"] = [
        {
            "name": "bad",
            "quota": 1,
            "fixed_elements": {
                "promoter_constraints": [{"upstream": "TTGACA", "downstream": "TATAAT", "spacer_length": [-1, 18]}]
            },
        }
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_side_biases_overlap_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["plan"] = [
        {
            "name": "bad",
            "quota": 1,
            "fixed_elements": {"side_biases": {"left": ["TTGACA"], "right": ["TTGACA"]}},
        }
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_solver_strategy_approximate_requires_no_options(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["solver"] = {"backend": "CBC", "strategy": "approximate", "options": ["Threads=2"]}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_solver_backend_optional_for_approximate(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["solver"] = {"strategy": "approximate", "options": []}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    load_config(cfg_path)


def test_side_biases_invalid_motif_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["generation"]["plan"] = [
        {
            "name": "bad",
            "quota": 1,
            "fixed_elements": {"side_biases": {"left": ["TTGAZ"], "right": []}},
        }
    ]
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_plots_source_required_for_multi_sink(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"] = {
        "targets": ["usr", "parquet"],
        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
        "usr": {"dataset": "demo", "root": "usr_root", "chunk_size": 10, "allow_overwrite": False},
        "parquet": {"path": "outputs/tables/demo_parquet.parquet", "deduplicate": True, "chunk_size": 128},
    }
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_plots_source_must_be_target(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    cfg["densegen"]["output"] = {
        "targets": ["usr", "parquet"],
        "schema": {"bio_type": "dna", "alphabet": "dna_4"},
        "usr": {"dataset": "demo", "root": "usr_root", "chunk_size": 10, "allow_overwrite": False},
        "parquet": {"path": "outputs/tables/demo_parquet.parquet", "deduplicate": True, "chunk_size": 128},
    }
    cfg["plots"] = {"source": "csv", "out_dir": "plots"}
    cfg_path = _write(cfg, tmp_path / "cfg.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_output_outside_run_root_rejected(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["densegen"]["output"]["parquet"]["path"] = "../outside/parquet"
    cfg_path = _write(cfg, run_dir / "config.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_config_must_live_inside_run_root(tmp_path: Path) -> None:
    cfg = copy.deepcopy(MIN_CONFIG)
    run_dir = tmp_path / "run_root"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["densegen"]["run"]["root"] = str(run_dir)
    cfg_path = _write(cfg, tmp_path / "config.yaml")
    with pytest.raises(ConfigError):
        load_config(cfg_path)
