"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_plot_manifest_integrity.py

Test integrity contracts for manifest-declared plot data.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.plots.manifests import (
    build_plot_manifest,
    refresh_plot_manifest_freshness,
    verified_plot_tidy_csv,
)


def test_plot_manifest_digest_binds_tidy_csv(tmp_path: Path) -> None:
    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True)
    media_path = plot_root / "score_r0.png"
    tidy_path = plot_root / "score_r0.csv"
    media_path.write_bytes(b"png")
    tidy_path.write_text(
        "as_of_round,id,view__is_selected\n0,candidate-a,true\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        output_dir=plot_root,
        filename=media_path.name,
        saved_data_paths=[tidy_path],
        data_paths={},
        artifact_metadata={},
        run_id="r0",
        selection_view_id="view-a",
        rounds=[0],
    )

    manifest = build_plot_manifest(
        name="score",
        kind="scatter_score_vs_rank",
        params={},
        context=context,
        status="written",
        started_at="2026-07-16T00:00:00+00:00",
    )

    tidy_outputs = [entry for entry in manifest["outputs"] if entry["role"] == "tidy_csv"]
    assert tidy_outputs == [
        {
            "role": "tidy_csv",
            "path": str(tidy_path),
            "exists": True,
            "size_bytes": tidy_path.stat().st_size,
            "mtime_ns": tidy_path.stat().st_mtime_ns,
            "sha256": file_sha256(tidy_path),
        }
    ]

    tidy_path.write_text(
        "as_of_round,id,view__is_selected\n0,candidate-b,false\n",
        encoding="utf-8",
    )
    refreshed = refresh_plot_manifest_freshness(manifest)
    refreshed_tidy = [entry for entry in refreshed["outputs"] if entry["role"] == "tidy_csv"]

    assert refreshed_tidy[0]["sha256"] == tidy_outputs[0]["sha256"]
    assert refreshed_tidy[0]["size_bytes"] == tidy_path.stat().st_size


def test_verified_plot_tidy_csv_rejects_post_manifest_mutation(tmp_path: Path) -> None:
    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True)
    media_path = plot_root / "score_r0.png"
    tidy_path = plot_root / "score_r0.csv"
    media_path.write_bytes(b"png")
    tidy_path.write_text(
        "as_of_round,id,view__is_selected\n0,candidate-a,true\n",
        encoding="utf-8",
    )
    manifest = build_plot_manifest(
        name="score",
        kind="scatter_score_vs_rank",
        params={},
        context=SimpleNamespace(
            output_dir=plot_root,
            filename=media_path.name,
            saved_data_paths=[tidy_path],
            data_paths={},
            artifact_metadata={},
            run_id="r0",
            selection_view_id="view-a",
            rounds=[0],
        ),
        status="written",
        started_at="2026-07-16T00:00:00+00:00",
    )

    assert verified_plot_tidy_csv(manifest, plot_root=plot_root) == tidy_path.resolve()

    tidy_path.write_text(
        "as_of_round,id,view__is_selected\n0,candidate-b,false\n",
        encoding="utf-8",
    )
    with pytest.raises(OpalError, match="SHA-256 does not match"):
        verified_plot_tidy_csv(manifest, plot_root=plot_root)


def test_verified_plot_tidy_csv_rejects_paths_outside_campaign_plot_root(tmp_path: Path) -> None:
    plot_root = tmp_path / "campaign" / "outputs" / "plots"
    plot_root.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("id\ncandidate-a\n", encoding="utf-8")

    for declared_path in (outside, plot_root / "linked.csv"):
        if declared_path != outside:
            declared_path.symlink_to(outside)
        manifest = {
            "tidy_csv": str(declared_path),
            "outputs": [
                {
                    "role": "tidy_csv",
                    "path": str(declared_path),
                    "sha256": file_sha256(outside),
                }
            ],
        }

        with pytest.raises(OpalError, match="outside the campaign plot root"):
            verified_plot_tidy_csv(manifest, plot_root=plot_root)
