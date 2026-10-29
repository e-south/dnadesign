"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_retry_completion.py

Construct retry completion and immutable lineage regression tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from dnadesign.construct import run_from_config
from dnadesign.construct.src.contracts.errors import ValidationError
from dnadesign.construct.src.orchestration import runtime
from dnadesign.construct.src.persistence import write_session
from dnadesign.construct.src.persistence.usr_registry import _ensure_construct_registry
from dnadesign.seqledger import Dataset, load_sequence_views
from dnadesign.seqledger.src.datasets.lifecycle.write_session import DatasetWriteSession


def _campaign(tmp_path: Path) -> tuple[Path, Dataset, str]:
    root = tmp_path / "datasets"
    _ensure_construct_registry(root)
    inputs = Dataset(root, "inputs")
    inputs.init(source="fixture")
    parent_id = inputs.add_sequences(["A" * 40 + "C" * 40], bio_type="dna", alphabet="dna_4").ids[0]
    inputs.write_overlay(
        "usr_label",
        pa.table({"id": [parent_id], "usr_label__primary": ["reference"], "usr_label__aliases": [["ref"]]}),
    )
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "job": {
                    "id": "retry",
                    "mode": "normalize_anchor",
                    "input": {"source": {"kind": "usr", "root": str(root), "dataset": "inputs"}, "field": "sequence"},
                    "normalize_anchor": {
                        "product_kind": "analysis_window",
                        "target_length": 60,
                        "focal_selector": {"kind": "chain", "selectors": [{"kind": "sequence_offset", "offset_0": 40}]},
                        "over_length_policy": {"kind": "trim", "target_length": 60},
                        "output_sequence_view": {"create": True},
                    },
                    "output": {
                        "target": {"kind": "usr", "root": str(root), "dataset": "outputs"},
                        "on_conflict": "ignore",
                    },
                }
            }
        )
    )
    return config, Dataset(root, "outputs"), parent_id


@pytest.mark.parametrize("stage", ["construct", "derived", "usr_label", "views"])
@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize("retry_from_error", [False, True])
def test_retry_completes_every_persistence_stage(
    tmp_path: Path, monkeypatch, stage: str, after_write: bool, retry_from_error: bool
) -> None:
    config, output, parent_id = _campaign(tmp_path)
    if retry_from_error:
        config.write_text(config.read_text().replace("on_conflict: ignore", "on_conflict: error"))
    original_overlay = DatasetWriteSession.write_overlay
    original_views = write_session.write_sequence_views

    def interrupted_overlay(self, namespace, *args, **kwargs):
        if namespace == stage and not after_write:
            raise OSError("interrupted persistence")
        result = original_overlay(self, namespace, *args, **kwargs)
        if namespace == stage:
            raise OSError("interrupted persistence")
        return result

    def interrupted_views(*args, **kwargs):
        if after_write:
            original_views(*args, **kwargs)
        raise OSError("interrupted persistence")

    with monkeypatch.context() as fault:
        fault.setattr(DatasetWriteSession, "write_overlay", interrupted_overlay)
        if stage == "views":
            fault.setattr(write_session, "write_sequence_views", interrupted_views)
        with pytest.raises(OSError, match="interrupted persistence"):
            run_from_config(config)

    config.write_text(config.read_text().replace("on_conflict: error", "on_conflict: ignore"))
    result = run_from_config(config)
    assert result.records_written == 0
    assert result.records_skipped_existing == 1
    row = output.head(n=1).iloc[0]
    assert row["construct__input_id"] == parent_id
    assert row["derived__parent_id"] == parent_id
    assert bool(row["derived__analysis_only"]) is True
    assert row["usr_label__primary"] == "reference_core60"
    views = load_sequence_views(output)
    assert len(views) == 1
    assert all(view.parent_sequence_id == parent_id for view in views)

    before = {path: path.read_bytes() for path in output.dir.rglob("*") if path.is_file() and path.suffix != ".lock"}
    run_from_config(config)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "namespace, column, value",
    [
        ("construct", "construct__input_id", "another-parent"),
        ("derived", "derived__analysis_only", False),
        ("usr_label", "usr_label__primary", "another-label"),
    ],
)
def test_retry_rejects_conflicting_effective_metadata(tmp_path: Path, namespace, column, value) -> None:
    config, output, _ = _campaign(tmp_path)
    run_from_config(config)
    row_id = output.head(n=1).iloc[0]["id"]
    output.write_overlay_part(namespace, pa.table({"id": [row_id], column: [value]}), overwrite=True)
    with pytest.raises(ValidationError, match="different metadata"):
        run_from_config(config)
    assert output.head(n=1).iloc[0][column] == value


def test_ignore_rechecks_completion_after_competing_writer(tmp_path: Path, monkeypatch) -> None:
    config, output, parent_id = _campaign(tmp_path)
    original = runtime._write_output_records
    intervened = False

    def intervening_writer(*args, **kwargs):
        nonlocal intervened
        if not intervened:
            intervened = True
            run_from_config(config)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_write_output_records", intervening_writer)
    result = run_from_config(config)
    assert result.records_written == 0
    assert result.records_skipped_existing == 1
    assert output.head(n=2)["construct__input_id"].tolist() == [parent_id]


def test_error_policy_still_rejects_competing_writer(tmp_path: Path, monkeypatch) -> None:
    config, output, parent_id = _campaign(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["job"]["output"]["on_conflict"] = "error"
    config.write_text(yaml.safe_dump(payload))
    original = runtime._write_output_records
    intervened = False

    def intervening_writer(*args, **kwargs):
        nonlocal intervened
        if not intervened:
            intervened = True
            run_from_config(config)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_write_output_records", intervening_writer)
    with pytest.raises(ValidationError, match="already exist"):
        run_from_config(config)
    assert output.head(n=2)["construct__input_id"].tolist() == [parent_id]


def test_row_only_retry_rejects_another_derivation(tmp_path: Path) -> None:
    config, output, _ = _campaign(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["job"]["normalize_anchor"]["output_sequence_view"]["create"] = False
    config.write_text(yaml.safe_dump(payload))
    run_from_config(config)
    payload["job"]["id"] = "different-derivation"
    config.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError, match="different metadata"):
        run_from_config(config)
    assert output.head(n=1).iloc[0]["construct__job"] == "retry"
