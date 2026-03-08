"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_validate_command.py

Validation command hardening tests for infer CLI config parsing behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.infer.cli import app
from dnadesign.infer.src.runtime.capacity_planner import GpuDeviceInfo, GpuInventory

_RUNNER = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_config_rejects_unknown_fields_with_config_exit_code(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_extra.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
  typo_field: 123
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    output = (result.stdout or "").lower()
    assert "extra inputs are not permitted" in output or "extra_forbidden" in output


def test_validate_config_rejects_wrong_type_with_config_exit_code(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_type.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
  batch_size: not_an_int
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    output = (result.stdout or "").lower()
    assert "valid integer" in output


def test_validate_config_requires_explicit_path_or_cwd_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["validate", "config"])

    assert result.exit_code == 2
    output = result.stdout or ""
    assert "ConfigError:" in output
    assert "No config found." in output
    assert "Pass --config or place config.yaml in the current" in output


def test_validate_config_rejects_usr_ingest_path_field(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_usr_path.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo_dataset
      path: inputs/records.jsonl
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    assert "ingest.path is not allowed for source='usr'" in (result.stdout or "")


def test_validate_config_fails_capacity_for_40b_on_single_gpu(monkeypatch, tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "capacity_fail_40b.yaml",
        """
model:
  id: evo2_40b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
    )

    monkeypatch.setattr(
        "dnadesign.infer.src.cli.commands.validate.probe_gpu_inventory",
        lambda: GpuInventory(
            devices=(
                GpuDeviceInfo(
                    index=0,
                    name="L40S",
                    total_memory_gib=45.0,
                    compute_capability="8.9",
                ),
            )
        ),
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", cfg.as_posix()])

    assert result.exit_code == 3
    assert "CAPACITY_FAIL" in (result.stdout or "")


def test_validate_usr_registry_renders_exact_namespace_register_command(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root
    outputs:
      - id: ll_mean
        fn: evo2.log_likelihood
        format: float
      - id: logits_mean
        fn: evo2.logits
        format: list
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "namespace: infer" in output
    assert "root: /tmp/usr-root" in output
    assert (
        "columns: infer__evo2_7b__j1__ll_mean:float64,"
        "infer__evo2_7b__j1__logits_mean:list<float64>"
    ) in output
    assert (
        "uv run usr --root /tmp/usr-root namespace register infer --columns "
        "'infer__evo2_7b__j1__ll_mean:float64,"
        "infer__evo2_7b__j1__logits_mean:list<float64>'"
    ) in output


def test_validate_usr_registry_filters_to_selected_job(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry_filter.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root
    outputs:
      - id: ll_mean
        fn: evo2.log_likelihood
        format: float
    io:
      write_back: true
  - id: j2
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root
    outputs:
      - id: logits_mean
        fn: evo2.logits
        format: list
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix(), "--job", "j2"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "infer__evo2_7b__j2__logits_mean:list<float64>" in output
    assert "infer__evo2_7b__j1__ll_mean" not in output


def test_validate_usr_registry_fails_fast_on_mixed_usr_roots(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry_mixed_roots.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root-a
    outputs:
      - id: ll_mean
        fn: evo2.log_likelihood
        format: float
    io:
      write_back: true
  - id: j2
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root-b
    outputs:
      - id: logits_mean
        fn: evo2.logits
        format: list
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix()])

    assert result.exit_code == 2
    assert "All selected USR write-back jobs must use the same ingest.root." in (result.stdout or "")


def test_validate_usr_registry_fails_fast_on_unsupported_output_format(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry_tensor.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root
    outputs:
      - id: emb
        fn: evo2.embedding
        format: tensor
        params:
          layer: mid
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix()])

    assert result.exit_code == 2
    assert "USR registry spec only supports infer output formats: float, list" in (result.stdout or "")
