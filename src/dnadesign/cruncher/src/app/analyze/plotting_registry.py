"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plotting_registry.py

Track analysis plot artifacts and prepare the plot output directory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.artifacts.entries import artifact_entry
from dnadesign.cruncher.artifacts.layout import run_plots_dir

__all__ = ["_prepare_analysis_plot_dir", "_record_analysis_plot"]


def _record_analysis_plot(
    *,
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
    spec_key: str,
    output: Path,
    generated: bool,
    skip_reason: str | None,
    run_dir: Path,
) -> None:
    spec = next(spec for spec in PLOT_SPECS if spec.key == spec_key)
    try:
        rel_output = output.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"analysis plot output must be inside run dir: {output}") from exc
    plot_entries.append(
        {
            "key": spec.key,
            "label": spec.label,
            "group": spec.group,
            "description": spec.description,
            "requires": list(spec.requires),
            "outputs": [{"path": str(rel_output), "exists": output.exists()}],
            "generated": generated,
            "skipped": not generated,
            "skip_reason": skip_reason,
        }
    )
    if generated and output.exists():
        plot_artifacts.append(
            artifact_entry(
                output,
                run_dir,
                kind="plot",
                label=spec.label,
                stage="analysis",
            )
        )


def _prepare_analysis_plot_dir(run_dir: Path) -> None:
    plots_dir = run_plots_dir(run_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy nested analysis plot directory from previous layout versions.
    legacy_analysis_dir = plots_dir / "analysis"
    if legacy_analysis_dir.exists():
        shutil.rmtree(legacy_analysis_dir)

    # Rewrite analysis plot outputs on each run without touching logo files.
    for spec in PLOT_SPECS:
        for output in plots_dir.glob(f"{spec.key}.*"):
            if output.is_file():
                output.unlink()
