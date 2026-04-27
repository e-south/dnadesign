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

import dnadesign.baserender.src.config.cruncher_showcase_job as cruncher_showcase_job
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


def test_adapter_renderer_compatibility_is_enforced_at_config_boundary(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["adapter"] = {
        "kind": "sequence_evidence_map_v1",
        "columns": {},
        "policies": {},
    }
    payload["render"]["renderer"] = "sequence_rows"
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="input.adapter.kind.*render.renderer"):
        load_cruncher_showcase_job(job_path)


def test_adapter_alphabet_compatibility_is_enforced_at_config_boundary(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    payload["input"]["alphabet"] = "RNA"
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="input.adapter.kind.*input.alphabet"):
        load_cruncher_showcase_job(job_path)


def test_run_report_is_opt_in_when_run_block_is_omitted(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    del payload["run"]
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)

    assert job.run.emit_report is False
    assert job.run.report_path is None


def test_video_output_rejects_conflicting_explicit_size_and_aspect_ratio(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[
            {
                "kind": "video",
                "fmt": "mp4",
                "width_px": 100,
                "height_px": 55,
                "aspect": 2.0,
            }
        ],
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="outputs\\[0\\].aspect"):
        load_cruncher_showcase_job(job_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload["input"].__setitem__("limit", "many"), "input.limit must be int"),
        (
            lambda payload: payload["input"].__setitem__("sample", {"mode": "first_n", "n": "many"}),
            "input.sample.n must be int",
        ),
        (
            lambda payload: payload["input"].__setitem__("sample", {"mode": "random_rows", "n": 2, "seed": "seed"}),
            "input.sample.seed must be int",
        ),
        (lambda payload: payload["outputs"][0].__setitem__("fps", "fast"), "outputs\\[0\\].fps must be int"),
        (
            lambda payload: payload["outputs"][0].__setitem__("frames_per_record", "many"),
            "outputs\\[0\\].frames_per_record must be int",
        ),
        (
            lambda payload: payload["outputs"][0].__setitem__("pauses", {"r1": "slow"}),
            "outputs\\[0\\].pauses.r1 must be float",
        ),
        (
            lambda payload: payload["outputs"][0].__setitem__("width_px", "wide"),
            "outputs\\[0\\].width_px must be int",
        ),
        (
            lambda payload: payload["outputs"][0].__setitem__("height_px", "tall"),
            "outputs\\[0\\].height_px must be int",
        ),
        (
            lambda payload: payload["outputs"][0].__setitem__("total_duration", "long"),
            "outputs\\[0\\].total_duration must be float",
        ),
        (
            lambda payload: payload["outputs"][0].__setitem__("title_font_size", "large"),
            "outputs\\[0\\].title_font_size must be int",
        ),
    ],
)
def test_scalar_coercion_errors_are_schema_errors(tmp_path: Path, mutate, match: str) -> None:
    parquet = _make_input_parquet(tmp_path)
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "results",
        outputs=[{"kind": "video", "fmt": "mp4"}],
    )
    mutate(payload)
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match=match):
        load_cruncher_showcase_job(job_path)


def test_declared_render_contract_rejects_incompatible_renderer(tmp_path: Path) -> None:
    json_path = tmp_path / "input.json"
    json_path.write_text("[]")
    payload = {
        "version": 3,
        "contract": {"kind": "sequence_rows_render_v3"},
        "results_root": str(tmp_path / "results"),
        "input": {
            "kind": "json",
            "path": str(json_path),
            "adapter": {"kind": "sequence_evidence_map_v1", "columns": {}, "policies": {}},
            "alphabet": "DNA",
        },
        "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(SchemaError, match="contract.kind.*render.renderer"):
        load_cruncher_showcase_job(job_path)


def test_declared_render_contract_records_use_case_descriptor(tmp_path: Path) -> None:
    json_path = tmp_path / "input.json"
    json_path.write_text("[]")
    payload = {
        "version": 3,
        "contract": {"kind": "nucleotide_evidence_map_render_v3"},
        "results_root": str(tmp_path / "results"),
        "input": {
            "kind": "json",
            "path": str(json_path),
            "adapter": {"kind": "sequence_evidence_map_v1", "columns": {}, "policies": {}},
            "alphabet": "DNA",
        },
        "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)

    assert job.contract.kind == "nucleotide_evidence_map_render_v3"
    assert job.render.renderer == "nucleotide_evidence_map"


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


def test_inline_job_mapping_accepts_explicit_absolute_input_and_output_paths_outside_caller_root(
    tmp_path: Path,
) -> None:
    caller_root = tmp_path / "caller"
    caller_root.mkdir()
    parquet = _make_input_parquet(tmp_path / "input_root")
    output_path = (tmp_path / "render_root" / "render.png").resolve()
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=caller_root / "results",
        outputs=[{"kind": "images", "path": str(output_path), "fmt": "png"}],
    )

    job = load_sequence_rows_job_from_mapping(
        payload,
        caller_root=caller_root,
        source_name="inline_job.yaml",
    )

    assert job.path == (caller_root / "inline_job.yaml").resolve()
    assert job.input.path == parquet.resolve()
    images_output = next(output for output in job.outputs if output.kind == "images")
    assert images_output.path == output_path


def test_cassette_job_rejects_input_path_outside_owner_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "cassette_run"
    outside_input = _make_input_parquet(tmp_path / "outside")
    payload = densegen_job_payload(
        parquet_path=outside_input,
        results_root=run_dir,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml", payload)

    with pytest.raises(SchemaError, match="must stay within"):
        load_cruncher_showcase_job(job_path)


def test_cassette_job_rejects_output_path_outside_owner_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "cassette_run"
    input_path = _make_input_parquet(run_dir / "inputs")
    payload = densegen_job_payload(
        parquet_path=input_path,
        results_root=run_dir,
        outputs=[{"kind": "images", "path": str(tmp_path / "leak.png"), "fmt": "png"}],
    )
    job_path = write_job(run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml", payload)

    with pytest.raises(SchemaError, match="must stay within"):
        load_cruncher_showcase_job(job_path)


def test_packaged_job_helpers_detect_examples_and_owner_roots(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    (jobs_root / "example.yml").write_text("version: 3\n")

    owner_root = tmp_path / "cassette_run"
    baserender_jobs = owner_root / "baserender_jobs"
    baserender_jobs.mkdir(parents=True)
    job_path = baserender_jobs / "top_hits_duplex.job.yaml"

    missing_owner_job = tmp_path / "missing_owner" / "baserender_jobs" / "top_hits_duplex.job.yaml"

    assert cruncher_showcase_job._has_packaged_job_examples(jobs_root) is True
    assert cruncher_showcase_job._published_job_owner_root_from_job_path(job_path) == owner_root.resolve()
    assert cruncher_showcase_job._cassette_run_root_from_job_path(job_path) == owner_root.resolve()
    assert cruncher_showcase_job._published_job_owner_root_from_job_path(missing_owner_job) is None


def test_inline_mapping_allowed_roots_collect_absolute_paths_and_ignore_empty_entries(tmp_path: Path) -> None:
    input_path = _make_input_parquet(tmp_path / "inputs")
    selection_path = tmp_path / "selection.csv"
    selection_path.write_text("id\nr1\n")
    config_path = tmp_path / "motifs.yaml"
    config_path.write_text("cruncher:\n  pwms_info: {}\n")
    output_dir = tmp_path / "images"
    output_path = tmp_path / "render.png"
    report_path = tmp_path / "report.json"
    job_path = tmp_path / "job.yaml"

    roots = cruncher_showcase_job._inline_mapping_allowed_roots(
        {
            "results_root": "",
            "input": {
                "path": str(input_path),
                "adapter": {
                    "columns": {
                        "hits_path": str(input_path),
                        "config_path": str(config_path),
                    }
                },
            },
            "selection": {"path": str(selection_path)},
            "pipeline": {
                "plugins": [
                    {
                        "attach_motifs_from_config": {
                            "config_path": str(config_path),
                        }
                    }
                ]
            },
            "outputs": [
                "skip-me",
                {"dir": str(output_dir), "path": str(output_path)},
            ],
            "run": {"report_path": str(report_path)},
        },
        caller_scope=tmp_path,
        job_path=job_path,
    )

    assert input_path.resolve() in roots
    assert selection_path.resolve() in roots
    assert config_path.resolve() in roots
    assert output_dir.resolve() in roots
    assert output_path.resolve() in roots
    assert report_path.resolve() in roots


def test_resolve_job_path_prefers_packaged_jobs_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_root = tmp_path / "baserender"
    jobs_root = fake_root / "jobs"
    docs_root = fake_root / "docs" / "examples"
    jobs_root.mkdir(parents=True)
    docs_root.mkdir(parents=True)
    packaged_job = jobs_root / "demo.yaml"
    docs_job = docs_root / "demo.yaml"
    packaged_job.write_text("version: 3\n")
    docs_job.write_text("version: 3\n")
    monkeypatch.setattr(cruncher_showcase_job, "_baserender_root", lambda: fake_root)

    assert cruncher_showcase_job.resolve_job_path("demo") == packaged_job


def test_resolve_job_path_reports_packaged_search_space_when_jobs_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_root = tmp_path / "baserender"
    jobs_root = fake_root / "jobs"
    jobs_root.mkdir(parents=True)
    (jobs_root / "other.yaml").write_text("version: 3\n")
    monkeypatch.setattr(cruncher_showcase_job, "_baserender_root", lambda: fake_root)

    with pytest.raises(FileNotFoundError, match="jobs/ or docs/examples"):
        cruncher_showcase_job.resolve_job_path("missing")
