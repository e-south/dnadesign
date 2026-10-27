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
from dnadesign.usr import Dataset, SequenceViewRecord, ensure_sequence_contract_namespaces, write_sequence_views

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


def test_validate_config_rejects_rootless_usr_writeback_job(tmp_path: Path) -> None:
    bad = _write(
        tmp_path / "bad_usr_writeback_rootless.yaml",
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
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", bad.as_posix()])

    assert result.exit_code == 2
    assert "USR write-back jobs must set ingest.root explicitly." in (result.stdout or "")


def test_validate_sequence_view_completion_renders_json_plan(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    config = _write(
        tmp_path / "config.yaml",
        f"""
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: reference_views
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      collect_log_likelihood: false
      sequence_view_inputs:
        - dataset: reference_views
          root: {usr_root.as_posix()}
          view_selector:
            product_kind: analysis_window
          pooling:
            operation: core60_mean
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "sequence-view-completion", "--config", config.as_posix(), "--format", "json"],
    )

    assert result.exit_code == 0, result.stdout
    assert '"required_views":1' in (result.stdout or "")
    assert '"missing_vectors":2' in (result.stdout or "")
    assert '"missing_scalars":0' in (result.stdout or "")

    inventory_result = _RUNNER.invoke(
        app,
        [
            "validate",
            "sequence-view-completion",
            "--config",
            config.as_posix(),
            "--format",
            "json",
            "--mode",
            "inventory",
        ],
    )

    assert inventory_result.exit_code == 0, inventory_result.stdout
    assert '"required_views":1' in (inventory_result.stdout or "")
    assert '"missing_vectors":2' in (inventory_result.stdout or "")
    assert '"stale_vectors":0' in (inventory_result.stdout or "")


def test_validate_sequence_view_completion_thresholds_fail_before_batch_submission(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    config = _write(
        tmp_path / "config.yaml",
        f"""
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: reference_views
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      collect_log_likelihood: false
      sequence_view_inputs:
        - dataset: reference_views
          root: {usr_root.as_posix()}
          view_selector:
            product_kind: source_record
          pooling:
            operation: seq_mean
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(
        app,
        [
            "validate",
            "sequence-view-completion",
            "--config",
            config.as_posix(),
            "--format",
            "json",
            "--max-missing-products",
            "0",
        ],
    )

    assert result.exit_code == 1
    output = " ".join((result.stdout or "").split())
    assert "sequence-view completion thresholds failed" in output
    assert "missing_products=1 exceeds max_missing_products=0" in output


def test_validate_sequence_view_completion_scalar_thresholds_fail_before_batch_submission(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    config = _write(
        tmp_path / "config.yaml",
        f"""
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: reference_views
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      collect_log_likelihood: true
      collect_output_layer_mean: false
      collect_intermediate_embedding: false
      sequence_view_inputs:
        - dataset: reference_views
          root: {usr_root.as_posix()}
          view_selector:
            product_kind: analysis_window
          pooling:
            operation: core60_mean
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(
        app,
        [
            "validate",
            "sequence-view-completion",
            "--config",
            config.as_posix(),
            "--format",
            "json",
            "--max-missing-scalars",
            "0",
        ],
    )

    assert result.exit_code == 1
    output = " ".join((result.stdout or "").split())
    assert "sequence-view completion thresholds failed" in output
    assert "missing_scalars=2 exceeds max_missing_scalars=0" in output


def test_validate_sequence_view_completion_resolves_roots_relative_to_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace" / "infer"
    workspace.mkdir(parents=True)
    usr_root = tmp_path / "workspace" / "usr_root"
    ensure_sequence_contract_namespaces(usr_root)
    dataset = Dataset(usr_root, "reference_views")
    dataset.init(source="test")
    add_result = dataset.add_sequences(["ACGT" * 15], bio_type="dna", alphabet="dna_4", source="test")
    write_sequence_views(
        dataset,
        [
            SequenceViewRecord(
                sequence_id=add_result.ids[0],
                view_name="core60_view",
                product_kind="analysis_window",
                context_kind="analysis_window",
                orientation="forward",
                analysis_only=True,
                source_dataset_id=dataset.name,
                anchor_start_0=0,
                anchor_end_0=60,
                recommended_pooling="core60_mean",
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        ],
        conflict_policy="error",
    )
    config = _write(
        workspace / "config.yaml",
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: reference_views
    operation: extract
    ingest:
      source: records
      field: sequence
    feature_bundle:
      collect_log_likelihood: false
      sequence_view_inputs:
        - dataset: reference_views
          root: ../usr_root
          view_selector:
            product_kind: analysis_window
          pooling:
            operation: core60_mean
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(
        app,
        ["validate", "sequence-view-completion", "--config", config.as_posix(), "--format", "json"],
    )

    assert result.exit_code == 0, result.stdout
    assert '"required_views":1' in (result.stdout or "")
    assert '"missing_products":0' in (result.stdout or "")


def test_validate_config_fails_capacity_for_20b_on_single_hopper_with_insufficient_memory(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _write(
        tmp_path / "capacity_fail_20b_hopper_memory.yaml",
        """
model:
  id: evo2_20b
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
                    name="H100",
                    total_memory_gib=45.0,
                    compute_capability="9.0",
                ),
            )
        ),
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", cfg.as_posix()])

    assert result.exit_code == 3
    assert "CAPACITY_FAIL" in (result.stdout or "")


def test_validate_config_rejects_20b_on_non_hopper_gpu(monkeypatch, tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "capacity_fail_20b_non_hopper.yaml",
        """
model:
  id: evo2_20b
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
                    total_memory_gib=80.0,
                    compute_capability="8.9",
                ),
            )
        ),
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", cfg.as_posix()])

    assert result.exit_code == 3
    assert "requires Hopper" in (result.stdout or "")


def test_validate_config_rejects_unsupported_feature_bundle_model(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "feature_bundle_model_contract.yaml",
        """
model:
  id: evo2_1b_base
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
    feature_bundle:
      context:
        kind: anchor_only
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "config", "--config", cfg.as_posix()])

    assert result.exit_code == 2
    assert "supports model.id values" in (result.stdout or "")


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
    resolved_root = Path("/tmp/usr-root").resolve().as_posix()
    assert "namespace: infer" in output
    assert f"root: {resolved_root}" in output
    assert ("columns: infer__evo2_7b__j1__ll_mean:float64,infer__evo2_7b__j1__logits_mean:list<float64>") in output
    assert (
        f"uv run usr --root {resolved_root} namespace register infer --columns "
        "'infer__evo2_7b__j1__ll_mean:float64,"
        "infer__evo2_7b__j1__logits_mean:list<float64>'"
    ) in output


def test_validate_usr_registry_includes_feature_bundle_metadata_columns(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry_feature_bundle.yaml",
        """
model:
  id: evo2_20b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: promoter_bundle
    operation: extract
    ingest:
      source: usr
      dataset: demo
      root: /tmp/usr-root
    feature_bundle:
      context:
        kind: anchor_only
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "infer__evo2_20b__promoter_bundle__log_likelihood__total:float64" in output
    assert "infer__evo2_20b__promoter_bundle__output_layer_mean__seq_mean:list<float64>" in output
    assert "infer__evo2_20b__promoter_bundle__intermediate_embedding__block23_mlp_out__seq_mean:list<float64>" in output
    assert "infer__evo2_20b__promoter_bundle__metadata__context_id:string" in output
    assert "infer__evo2_20b__promoter_bundle__metadata__is_wildtype:bool" in output
    assert "infer__evo2_20b__promoter_bundle__metadata__pooling_modes:list<string>" in output
    assert "infer__evo2_20b__promoter_bundle__metadata__feature_request_digest:string" in output


def test_validate_usr_registry_resolves_relative_ingest_root_from_config(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path / "usr_registry_relative_root.yaml",
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
      root: outputs/usr_datasets
    outputs:
      - id: ll_mean
        fn: evo2.log_likelihood
        format: float
    io:
      write_back: true
""".strip()
        + "\n",
    )

    result = _RUNNER.invoke(app, ["validate", "usr-registry", "--config", cfg.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert f"root: {(tmp_path / 'outputs' / 'usr_datasets').resolve()}" in (result.stdout or "")


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
