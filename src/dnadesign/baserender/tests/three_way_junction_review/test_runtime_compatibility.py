"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/three_way_junction_review/test_runtime_compatibility.py

Runtime compatibility checks for already-typed three-way-junction render jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import VideoOutputCfg

from .fixtures import _review_job


def _typed_review_job(tmp_path: Path, *, contract_kind: str) -> baserender.RenderJobV4:
    source = tmp_path / "three_way_junction_review.v1.json"
    source.write_text("[]", encoding="utf-8")
    job = baserender.validate_job(
        _review_job(source, contract_kind=contract_kind),
        caller_root=tmp_path,
    )
    source.unlink()
    return job


def _typed_generic_job(tmp_path: Path) -> baserender.RenderJobV4:
    source = tmp_path / "generic.json"
    source.write_text('[{"sequence":"ACGT","features":[]}]', encoding="utf-8")
    job = baserender.validate_job(
        {
            "version": 4,
            "contract": {"kind": "render_job_v4"},
            "bundle": {"path": "generic-render"},
            "input": {
                "kind": "json",
                "path": source.name,
                "adapter": {
                    "kind": "generic_features",
                    "columns": {"sequence": "sequence", "features": "features"},
                },
                "alphabet": "DNA",
            },
            "render": {"renderer": "sequence_rows", "style": {}},
            "outputs": [{"kind": "images", "dir": "images", "fmt": "svg"}],
            "run": {"strict": True, "fail_on_skips": True},
        },
        caller_root=tmp_path,
    )
    source.unlink()
    return job


def test_runtime_rejects_typed_adapter_renderer_mismatch_before_source_read(tmp_path: Path) -> None:
    job = _typed_review_job(tmp_path, contract_kind="render_job_v4")
    forged = replace(job, render=replace(job.render, renderer="sequence_rows"))

    with pytest.raises(
        baserender.SchemaError,
        match=(
            "input.adapter.kind 'three_way_junction_review_v1' is not compatible with render.renderer 'sequence_rows'"
        ),
    ):
        baserender.run_job(forged)

    assert not (tmp_path / "review-render").exists()


def test_runtime_rejects_typed_adapter_alphabet_mismatch_before_source_read(tmp_path: Path) -> None:
    job = _typed_review_job(tmp_path, contract_kind="render_job_v4")
    forged = replace(job, input=replace(job.input, alphabet="IUPAC_DNA"))

    with pytest.raises(
        baserender.SchemaError,
        match=("input.adapter.kind 'three_way_junction_review_v1' is not compatible with input.alphabet 'IUPAC_DNA'"),
    ):
        baserender.run_job(forged)

    assert not (tmp_path / "review-render").exists()


def test_runtime_rejects_typed_contract_renderer_mismatch_before_source_read(tmp_path: Path) -> None:
    job = _typed_review_job(tmp_path, contract_kind="three_way_junction_review_render_v1")
    generic_adapter = replace(
        job.input.adapter,
        kind="generic_features",
        columns={"sequence": "sequence", "features": "features"},
    )
    forged = replace(
        job,
        input=replace(job.input, adapter=generic_adapter),
        render=replace(job.render, renderer="sequence_rows"),
    )

    with pytest.raises(
        baserender.SchemaError,
        match=(
            "contract.kind 'three_way_junction_review_render_v1' is not compatible with render.renderer 'sequence_rows'"
        ),
    ):
        baserender.run_job(forged)

    assert not (tmp_path / "review-render").exists()


@pytest.mark.parametrize(
    "mutation",
    ["empty", "missing_destination", "outside_bundle", "single_file", "video"],
)
def test_runtime_rejects_forged_typed_outputs_before_source_read(tmp_path: Path, mutation: str) -> None:
    job = _typed_review_job(tmp_path, contract_kind="render_job_v4")
    image_output = job.outputs[0]
    if mutation == "empty":
        outputs = ()
        error = "outputs must contain at least one output entry"
    elif mutation == "missing_destination":
        outputs = (replace(image_output, dir=None, path=None),)
        error = "must define exactly one of dir or path"
    elif mutation == "outside_bundle":
        outputs = (replace(image_output, dir=tmp_path / "outside"),)
        error = "must stay inside bundle.path"
    elif mutation == "single_file":
        outputs = (replace(image_output, dir=None, path=job.bundle.path / "review.svg"),)
        error = "requires a directory for images output"
    else:
        outputs = (
            VideoOutputCfg(
                kind="video",
                path=job.bundle.path / "review.mp4",
                fmt="mp4",
                fps=1,
                frames_per_record=1,
                pauses={},
                width_px=100,
                height_px=100,
                aspect_ratio=None,
                total_duration=None,
            ),
        )
        error = "only supports output kinds: images"

    with pytest.raises(baserender.SchemaError, match=error):
        baserender.run_job(replace(job, outputs=outputs))

    assert not (tmp_path / "review-render").exists()


@pytest.mark.parametrize("output_kind", ["images", "video"])
def test_runtime_rejects_file_destination_equal_to_bundle_before_source_read(
    tmp_path: Path,
    output_kind: str,
) -> None:
    job = _typed_generic_job(tmp_path)
    if output_kind == "images":
        outputs = (replace(job.outputs[0], dir=None, path=job.bundle.path),)
    else:
        outputs = (
            VideoOutputCfg(
                kind="video",
                path=job.bundle.path,
                fmt="mp4",
                fps=1,
                frames_per_record=1,
                pauses={},
                width_px=100,
                height_px=100,
                aspect_ratio=None,
                total_duration=None,
            ),
        )

    with pytest.raises(baserender.SchemaError, match="must name a file inside bundle.path"):
        baserender.run_job(replace(job, outputs=outputs))

    assert not job.bundle.path.exists()


def test_runtime_rejects_filesystem_root_as_bundle_before_source_read(tmp_path: Path) -> None:
    job = _typed_generic_job(tmp_path)
    forged = replace(
        job,
        bundle=replace(job.bundle, path=Path("/")),
        outputs=(replace(job.outputs[0], dir=Path("/forged-render-output")),),
    )

    with pytest.raises(baserender.SchemaError, match="bundle.path must name an owned directory"):
        baserender.run_job(forged)

    assert not job.bundle.path.exists()
