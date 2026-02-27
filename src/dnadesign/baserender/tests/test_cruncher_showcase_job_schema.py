"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_cruncher_showcase_job_schema.py

Tests for cruncher showcase job strict schema validation behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.baserender.src.config import load_cruncher_showcase_job, load_sequence_rows_job_from_mapping
from dnadesign.baserender.src.core import SchemaError

from .conftest import densegen_job_payload, write_job, write_parquet


def _make_input_parquet(tmp_path: Path) -> Path:
    return write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            }
        ],
    )


def test_unknown_top_level_key_raises_schema_error(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
        extra={"unknown_top": 123},
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="Unknown keys in top-level"):
        load_cruncher_showcase_job(job_path)


def test_unknown_adapter_columns_key_raises_schema_error(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["adapter"]["columns"]["unexpected"] = "bad"
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="Unknown keys in input.adapter.columns"):
        load_cruncher_showcase_job(job_path)


def test_unknown_densegen_policy_key_raises_schema_error(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["adapter"]["policies"]["typo_policy"] = True
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="Unknown keys in input.adapter.policies"):
        load_cruncher_showcase_job(job_path)


def test_unknown_generic_features_policy_key_raises_schema_error(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [
                    {
                        "id": "k1",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "ACGT",
                        "tags": ["tf:x"],
                    }
                ],
            }
        ],
    )
    payload = {
        "version": 3,
        "results_root": str(tmp_path / "results"),
        "input": {
            "kind": "parquet",
            "path": str(parquet),
            "adapter": {
                "kind": "generic_features",
                "columns": {
                    "sequence": "sequence",
                    "features": "features",
                    "id": "id",
                },
                "policies": {"typo_policy": "x"},
            },
            "alphabet": "DNA",
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="Unknown keys in input.adapter.policies"):
        load_cruncher_showcase_job(job_path)


def test_selection_keep_order_requires_bool(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    selection_csv = tmp_path / "selection.csv"
    selection_csv.write_text("id\nr1\n")
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
        extra={
            "selection": {
                "path": str(selection_csv),
                "match_on": "id",
                "column": "id",
                "overlay_column": None,
                "keep_order": "false",
                "on_missing": "error",
            }
        },
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="selection.keep_order must be bool"):
        load_cruncher_showcase_job(job_path)


def test_run_flags_require_bool(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["run"] = {"strict": "true", "fail_on_skips": False, "emit_report": True}
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="run.strict must be bool"):
        load_cruncher_showcase_job(job_path)


def test_run_must_be_mapping_when_provided(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["run"] = []
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="run must be a mapping"):
        load_cruncher_showcase_job(job_path)


def test_densegen_bool_policies_require_bool_type(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["adapter"]["policies"]["zero_as_unspecified"] = "false"
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="input.adapter.policies.zero_as_unspecified must be bool"):
        load_cruncher_showcase_job(job_path)


def test_default_results_root_scopes_to_job_directory(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "ignored_results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    del payload["results_root"]
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)
    assert job.results_root == (tmp_path / "results").resolve()


def test_named_example_job_default_results_root_is_job_local(tmp_path: Path) -> None:
    job = load_cruncher_showcase_job("densegen_job")
    assert job.path.name == "densegen_job.yaml"
    assert job.results_root == (job.path.parent / "results").resolve()


def test_explicit_caller_root_overrides_default_results_scope(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "ignored_results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    del payload["results_root"]
    job_path = write_job(tmp_path / "job.yaml", payload)

    caller_root = tmp_path / "caller"
    caller_root.mkdir()
    job = load_cruncher_showcase_job(job_path, caller_root=caller_root)
    assert job.results_root == (caller_root / "results").resolve()


def test_absolute_input_path_must_exist_at_config_boundary(tmp_path: Path) -> None:
    missing_input = (tmp_path / "missing.parquet").resolve()
    payload = densegen_job_payload(
        parquet_path=tmp_path / "placeholder.parquet",
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["path"] = str(missing_input)
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="input.path does not exist"):
        load_cruncher_showcase_job(job_path)


def test_attach_motifs_plugin_path_resolves_at_config_boundary(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    motif_cfg = tmp_path / "config_used.yaml"
    motif_cfg.write_text("cruncher:\n  pwms_info: {}\n")
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["pipeline"] = {
        "plugins": [
            {
                "attach_motifs_from_config": {
                    "config_path": "config_used.yaml",
                    "require_effect": False,
                }
            }
        ]
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)
    plugin = job.pipeline.plugins[0]
    assert plugin.name == "attach_motifs_from_config"
    assert Path(str(plugin.params["config_path"])) == motif_cfg.resolve()


def test_attach_motifs_from_library_path_resolves_at_config_boundary(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    library = tmp_path / "motif_library.json"
    library.write_text(
        (
            "{\n"
            '  "schema_version": "1",\n'
            '  "alphabet": "DNA",\n'
            '  "motifs": {\n'
            '    "lexA": {\n'
            '      "source": "demo",\n'
            '      "motif_id": "lexA_demo",\n'
            '      "matrix": [[0.1, 0.8, 0.05, 0.05]]\n'
            "    }\n"
            "  }\n"
            "}\n"
        )
    )
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["pipeline"] = {
        "plugins": [
            {
                "attach_motifs_from_library": {
                    "library_path": "motif_library.json",
                    "require_effect": False,
                }
            }
        ]
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)
    plugin = job.pipeline.plugins[0]
    assert plugin.name == "attach_motifs_from_library"
    assert Path(str(plugin.params["library_path"])) == library.resolve()


def test_inline_job_source_name_rejects_directory_components(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=Path("input.parquet"),
        results_root=Path("results"),
        outputs=[{"kind": "images", "fmt": "png"}],
    )

    with pytest.raises(SchemaError, match="source_name must be a simple filename"):
        load_sequence_rows_job_from_mapping(
            payload,
            caller_root=tmp_path,
            source_name="nested/job.yaml",
        )

    # Baseline behavior remains explicit and valid with a plain filename.
    job = load_sequence_rows_job_from_mapping(
        payload,
        caller_root=tmp_path,
        source_name="inline_job.yaml",
    )
    assert job.path == (tmp_path / "inline_job.yaml").resolve()
    assert job.input.path == parquet.resolve()
