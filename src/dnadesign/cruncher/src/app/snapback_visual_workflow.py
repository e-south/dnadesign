"""
Application orchestration for visual-only Snapback examples.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text
from dnadesign.cruncher.snapback.load import load_snapback_visual_spec
from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.visual_models import SnapbackVisualReport
from dnadesign.cruncher.snapback.visual_plot import render_snapback_visual_plot


def run_snapback_visual(path: str | Path, *, force_overwrite: bool = False) -> tuple[Path, SnapbackVisualReport]:
    spec, spec_path, workspace_root = load_snapback_visual_spec(path)
    run_dir = workspace_root / spec.output.run_dir
    if run_dir.exists():
        if not force_overwrite:
            raise ValueError(f"Snapback visual run directory already exists: {run_dir}. Use --force-overwrite.")
        shutil.rmtree(run_dir)

    analysis_dir = run_dir / "analysis"
    plots_dir = run_dir / "plots"
    meta_dir = run_dir / "meta"
    provenance_dir = run_dir / "provenance"
    for directory in (analysis_dir, plots_dir, meta_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    plot_path = plots_dir / f"{spec.name}.snapback_visual.{spec.output.render_format}"
    plot_data = render_snapback_visual_plot(spec, plot_path)
    plot_data_path = analysis_dir / "snapback_visual_plot_data.json"
    atomic_write_json(plot_data_path, plot_data)
    atomic_write_text(provenance_dir / "spec.snapshot.yaml", spec_path.read_text(encoding="utf-8"))

    report = SnapbackVisualReport(
        spec_name=spec.name,
        workspace_root=str(workspace_root.resolve()),
        spec_path=str(spec_path.resolve()),
        run_dir=str(run_dir.resolve()),
        plot_data_path=str(plot_data_path.resolve()),
        plot_path=str(plot_path.resolve()),
        precursor_top_strand=spec.input.precursor_top_strand,
        precursor_bottom_strand=complement_sequence(spec.input.precursor_top_strand),
        nick_label=spec.nick.label,
        nick_boundary_from_left=spec.nick.nick_boundary,
        active_product_sequence=spec.active_product_sequence,
        upstream_context_nt=spec.product.upstream_context_nt,
        effective_stem_bp=spec.effective_stem_bp,
        stem_sequence=spec.product.stem_sequence,
        cap_sequence=spec.product.cap_sequence,
        foldback_sequence=spec.product.foldback_sequence,
    )
    atomic_write_json(analysis_dir / "snapback_visual_report.json", report.model_dump(mode="json"))
    atomic_write_json(
        meta_dir / "snapback_visual_manifest.json",
        {
            "kind": "snapback_visual_manifest_v1",
            "status": report.status,
            "spec_name": report.spec_name,
            "workspace_root": report.workspace_root,
            "spec_path": report.spec_path,
            "run_dir": report.run_dir,
            "artifacts": {
                "report": str((analysis_dir / "snapback_visual_report.json").resolve()),
                "plot_data_json": report.plot_data_path,
                "plot": report.plot_path,
                "spec_snapshot": str((provenance_dir / "spec.snapshot.yaml").resolve()),
            },
        },
    )
    atomic_write_json(
        meta_dir / "snapback_visual_status.json",
        {
            "kind": "snapback_visual_status_v1",
            "status": report.status,
            "spec_name": report.spec_name,
            "run_dir": report.run_dir,
            "plot_path": report.plot_path,
        },
    )
    return run_dir, report


__all__ = ["run_snapback_visual"]
