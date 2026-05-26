"""Configured OPAL plot review and quality checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from dnadesign.opal import (
    list_configured_plot_specs,
    load_plot_artifact_manifest,
    load_plot_config,
    load_plot_manifest_index,
)

from ..artifacts import ProbeArtifactLayout
from .configured_plot_files import (
    _expected_tidy_rounds_for_plot,
    _image_quality_problems,
    _plot_requires_tidy_csv,
    _tidy_csv_quality_problems,
)
from .configured_plot_scopes import _configured_spec_coverage_problems, _round_scope_coverage_problems


def _build_configured_plot_reviews(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    expected_rounds = {
        str(row.get("run_key")): int(row.get("as_of_round"))
        for row in metrics_payload.get("runs") or []
        if isinstance(row, Mapping) and row.get("run_key") and row.get("as_of_round") is not None
    }
    seen: set[str] = set()
    for row in metrics_payload.get("runs") or []:
        if not isinstance(row, Mapping):
            continue
        run_key = str(row.get("run_key") or "").strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        workdir = layout.campaign_workdir(run_key)
        plot_config = _load_current_plot_config(workdir)
        plots_dir = workdir / "outputs" / "plots"
        index_path = plots_dir / "plot_manifest.json"
        entry: dict[str, Any] = {
            "run_key": run_key,
            "plots_dir": str(plots_dir),
            "index_path": str(index_path),
            "expected_final_round": expected_rounds.get(run_key),
            "configured_plot_config": {key: value for key, value in plot_config.items() if key != "specs"},
            "expected_configured_plot_specs": plot_config.get("specs") or [],
            "status": "missing_index",
            "plot_count": 0,
            "plots": [],
            "quality": {"status": "missing", "problems": [f"configured_plot_index_missing:{run_key}"]},
        }
        if not index_path.exists():
            reviewed.append(entry)
            continue
        try:
            index = load_plot_manifest_index(index_path)
            plots = [_configured_plot_entry(plot_row) for plot_row in index.get("manifests") or []]
            entry.update(
                {
                    "status": "loaded",
                    "plot_count": len(plots),
                    "generated_at": index.get("generated_at"),
                    "plots": plots,
                }
            )
            entry["quality"] = _quality_for_configured_plot_entry(entry)
        except Exception as exc:
            entry.update(
                {
                    "status": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "quality": {
                        "status": "error",
                        "problems": [f"plot_manifest_error:{type(exc).__name__}:{exc}"],
                    },
                }
            )
        reviewed.append(entry)
    return reviewed


def _configured_plot_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = row.get("manifest_path")
    if manifest_path and Path(str(manifest_path)).exists():
        manifest = load_plot_artifact_manifest(str(manifest_path))
    else:
        manifest = dict(row)
    outputs = [dict(output) for output in manifest.get("outputs") or [] if isinstance(output, Mapping)]
    media = [output for output in outputs if output.get("role") == "media"]
    tidy = [output for output in outputs if output.get("role") == "tidy_csv"]
    return {
        "name": manifest.get("name"),
        "kind": manifest.get("kind"),
        "status": manifest.get("status"),
        "generated_at": manifest.get("generated_at"),
        "run_id": manifest.get("run_id"),
        "rounds": manifest.get("rounds"),
        "manifest_path": manifest.get("manifest_path") or manifest_path,
        "media_paths": [str(output.get("path")) for output in media if output.get("path")],
        "tidy_csv_paths": [str(output.get("path")) for output in tidy if output.get("path")],
        "params": manifest.get("params") or {},
        "metadata": manifest.get("metadata") or {},
        "quality": manifest.get("quality") or {},
        "freshness": manifest.get("freshness"),
        "warnings": manifest.get("warnings") or [],
        "error": manifest.get("error"),
    }


def _quality_for_configured_plot_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    expected_final_round = entry.get("expected_final_round")
    expected_rounds = set(range(int(expected_final_round) + 1)) if expected_final_round is not None else set()
    plot_config = (
        entry.get("configured_plot_config") if isinstance(entry.get("configured_plot_config"), Mapping) else {}
    )
    if plot_config.get("status") == "error":
        problems.append(
            "configured_plot_config_error:"
            f"{plot_config.get('error_type', 'unknown')}:{plot_config.get('error_message', '')}"
        )
    problems.extend(
        _configured_spec_coverage_problems(
            entry.get("plots") or [],
            entry.get("expected_configured_plot_specs") or [],
            expected_final_round=expected_final_round,
        )
    )
    problems.extend(_round_scope_coverage_problems(entry.get("plots") or [], expected_final_round=expected_final_round))
    for plot in entry.get("plots") or []:
        if not isinstance(plot, Mapping):
            problems.append("plot_entry_not_mapping")
            continue
        name = str(plot.get("name") or "unknown")
        if plot.get("status") != "written":
            problems.append(f"{name}:status_not_written")
        freshness = plot.get("freshness")
        if isinstance(freshness, Mapping) and freshness.get("status") != "fresh":
            problems.append(f"{name}:freshness_not_fresh:{freshness.get('status', 'unknown')}")
        media_paths = [Path(str(path)) for path in plot.get("media_paths") or []]
        if not media_paths:
            problems.append(f"{name}:media_missing")
        for media_path in media_paths:
            problems.extend(_image_quality_problems(media_path, label=name))
        tidy_paths = [Path(str(path)) for path in plot.get("tidy_csv_paths") or []]
        if not tidy_paths:
            if _plot_requires_tidy_csv(plot):
                problems.append(f"{name}:tidy_csv_missing")
            continue
        plot_expected_rounds = _expected_tidy_rounds_for_plot(
            plot.get("rounds"),
            expected_final_round=expected_final_round,
        )
        for tidy_path in tidy_paths:
            problems.extend(
                _tidy_csv_quality_problems(
                    tidy_path,
                    label=name,
                    kind=str(plot.get("kind") or ""),
                    expected_rounds=plot_expected_rounds if plot_expected_rounds is not None else expected_rounds,
                )
            )
    return {
        "status": "ok" if not problems else "attention",
        "problems": problems,
    }


def _load_current_plot_config(workdir: Path) -> dict[str, Any]:
    campaign_yaml = workdir / "configs" / "campaign.yaml"
    if not campaign_yaml.exists():
        return {
            "status": "missing_campaign_yaml",
            "campaign_yaml": str(campaign_yaml),
            "specs": [],
        }
    try:
        raw = yaml.safe_load(campaign_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"campaign YAML did not parse to a mapping: {campaign_yaml}")
        if not _campaign_declares_plot_config(raw):
            return {
                "status": "not_configured",
                "campaign_yaml": str(campaign_yaml),
                "specs": [],
            }
        plot_config = load_plot_config(
            campaign_cfg=raw,
            campaign_yaml=campaign_yaml,
            campaign_dir=workdir,
            plot_config_opt=None,
        )
        specs = list_configured_plot_specs(
            plots_cfg=plot_config.plots,
            plot_presets=plot_config.plot_presets,
        )
    except Exception as exc:
        return {
            "status": "error",
            "campaign_yaml": str(campaign_yaml),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "specs": [],
        }
    return {
        "status": "loaded",
        "campaign_yaml": str(campaign_yaml),
        "source_path": str(plot_config.source_path),
        "source_label": plot_config.source_label,
        "spec_count": len(specs),
        "specs": [dict(spec) for spec in specs if spec.get("enabled") is not False],
    }


def _campaign_declares_plot_config(raw: Mapping[str, Any]) -> bool:
    return any(key in raw for key in ("plots", "plot_defaults", "plot_presets", "plot_config"))


def _plot_quality_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    problems = [
        {"run_key": entry.get("run_key"), "problem": problem}
        for entry in entries
        for problem in ((entry.get("quality") or {}).get("problems") or [])
    ]
    loaded = [entry for entry in entries if entry.get("status") == "loaded"]
    return {
        "status": "ok" if not problems else "attention",
        "campaigns_with_plot_index": len(loaded),
        "campaigns_expected": len(entries),
        "plot_count": sum(int(entry.get("plot_count") or 0) for entry in loaded),
        "problem_count": len(problems),
        "problems": problems,
    }


def _review_next_steps(*, layout: ProbeArtifactLayout, plot_quality: Mapping[str, Any]) -> dict[str, str]:
    if plot_quality.get("status") == "ok":
        return {}
    module = "dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe"
    run_root = str(layout.run_root)
    return {
        "configured_plot_refresh_command": (f"uv run python -m {module} plot --run-root {run_root} --round all --json"),
        "rerun_report_command": f"uv run python -m {module} report --run-root {run_root} --plots --json",
    }
