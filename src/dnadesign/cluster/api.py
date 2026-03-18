"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/api.py

Public cluster execution helpers.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .src.execution import run_analyze as _run_analyze
from .src.execution import run_fit as _run_fit
from .src.execution import run_sweep as _run_sweep
from .src.execution import run_umap as _run_umap
from .src.layout import ClusterLayoutError, explicit_results_root
from .src.runs.contracts import fit_alias_from_cluster_col
from .src.runs.index import list_runs as _list_runs
from .src.workspaces import WorkspaceConfig, load_workspace_config


class ClusterApiError(RuntimeError):
    """Raised when a public cluster API command fails."""


@dataclass(frozen=True, slots=True)
class ClusterExecutionResult:
    command: Literal["fit", "umap", "analyze", "sweep"]
    subject: str
    results_root: Path
    artifact_path: Path
    workspace_id: str | None = None
    workspace_dir: Path | None = None
    run_record: dict[str, Any] | None = None


def _workspace_context(workspace: str | Path) -> WorkspaceConfig:
    return load_workspace_config(workspace)


def _resolved_results_root(results_root: str | Path) -> Path:
    try:
        return explicit_results_root(results_root)
    except ClusterLayoutError as exc:
        raise ClusterApiError(str(exc)) from exc


def _normalized_csv(values: str | Sequence[Any] | None) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    tokens = [str(value).strip() for value in values if str(value).strip()]
    return ",".join(tokens) if tokens else None


def _normalized_list(values: str | Sequence[Any] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _merged_workspace_section(
    config: WorkspaceConfig,
    section: Literal["fit", "umap", "analyze"],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = config.section_params(section)
    plot = config.section_plot(section)
    if not overrides:
        return params, plot
    merged = dict(params)
    plot_cfg = dict(plot)
    for key, value in overrides.items():
        if key == "plot" and isinstance(value, Mapping):
            plot_cfg.update(dict(value))
            continue
        if key == "method_params" and isinstance(value, Mapping):
            base = dict(merged.get("method_params", {}) or {})
            base.update(dict(value))
            merged["method_params"] = base
            continue
        merged[key] = value
    return merged, plot_cfg


def _run_record_for_alias(root: Path, command: str, alias: str | None) -> dict[str, Any] | None:
    kind = "analysis" if command == "analyze" else command
    runs = _list_runs(
        root=root,
        filters={"kind": kind, "alias": alias} if alias else {"kind": kind},
    )
    if runs.empty:
        return None
    return runs.iloc[0].to_dict()


def _run_record_for_artifact(
    root: Path,
    command: Literal["fit", "umap", "analyze", "sweep"],
    artifact_path: Path,
    *,
    alias: str | None = None,
) -> dict[str, Any] | None:
    kind = "analysis" if command == "analyze" else command
    filters: dict[str, Any] = {"kind": kind}
    if alias:
        filters["alias"] = alias
    runs = _list_runs(root=root, filters=filters)
    if runs.empty:
        return None

    artifact = artifact_path.resolve()
    column = {
        "fit": "labels_path",
        "umap": "plot_paths",
        "analyze": "analysis_path",
        "sweep": "sweep_path",
    }[command]
    if column not in runs.columns:
        return None

    def _matches(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            return False
        path = Path(text)
        candidate = path if command == "umap" else path.parent
        return candidate.resolve() == artifact

    if command == "umap":

        def _matches_umap_coords(value: Any) -> bool:
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "<na>"}:
                return False
            return _matches(Path(text).parent)

        matched = runs[runs["plot_paths"].apply(_matches) | runs["coords_path"].apply(_matches_umap_coords)]
    else:
        matched = runs[runs[column].apply(_matches)]
    return matched.iloc[0].to_dict() if not matched.empty else None


def list_runs(results_root: str | Path):
    return _list_runs(root=_resolved_results_root(results_root))


def list_workspace_runs(workspace: str | Path):
    config = _workspace_context(workspace)
    return list_runs(config.results_root)


def run_fit(
    *,
    results_root: str | Path,
    dataset: str | None = None,
    file: str | Path | None = None,
    usr_root: str | Path | None = None,
    name: str | None = None,
    key_col: str = "id",
    x_col: str | None = None,
    x_cols: str | Sequence[str] | None = None,
    method: str = "leiden",
    preset: str | None = None,
    method_params: Mapping[str, Any] | None = None,
    silhouette: bool = False,
    full_silhouette: bool = False,
    dedupe_policy: str = "error",
    reuse: str = "auto",
    force: bool = False,
    write: bool = False,
    allow_overwrite: bool = False,
    inplace: bool = False,
    out: str | Path | None = None,
    workspace_id: str | None = None,
    workspace_dir: str | Path | None = None,
) -> ClusterExecutionResult:
    root = _resolved_results_root(results_root)
    try:
        execution = _run_fit(
            dataset=dataset,
            file=str(file) if file is not None else None,
            usr_root=str(usr_root) if usr_root is not None else None,
            name=name,
            key_col=key_col,
            x_col=x_col,
            x_cols=_normalized_csv(x_cols),
            method=method,
            preset=preset,
            method_params=dict(method_params or {}),
            silhouette=bool(silhouette),
            full_silhouette=bool(full_silhouette),
            dedupe_policy=dedupe_policy,
            reuse=reuse,
            force=bool(force),
            write=bool(write),
            allow_overwrite=bool(allow_overwrite),
            inplace=bool(inplace),
            out=str(out) if out is not None else None,
            root=root,
            workspace_id=workspace_id,
            console=None,
        )
    except Exception as exc:
        raise ClusterApiError(f"cluster fit failed: {exc}") from exc
    return ClusterExecutionResult(
        command="fit",
        subject=execution.subject,
        results_root=root,
        artifact_path=execution.artifact_path,
        workspace_id=workspace_id,
        workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
        run_record=_run_record_for_artifact(
            root,
            "fit",
            execution.artifact_path,
            alias=execution.run_record_subject,
        ),
    )


def run_umap(
    *,
    results_root: str | Path,
    dataset: str | None = None,
    file: str | Path | None = None,
    usr_root: str | Path | None = None,
    name: str,
    key_col: str = "id",
    x_col: str | None = None,
    x_cols: str | Sequence[str] | None = None,
    neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str | None = None,
    random_state: int | None = None,
    preset: str | None = None,
    color_by: Sequence[str] | None = None,
    highlight: str | Path | None = None,
    highlight_topn: int | None = None,
    highlight_topn_col: str | None = None,
    highlight_topn_asc: bool = False,
    highlight_hue_col: str | None = None,
    alpha: float | None = None,
    size: float | None = None,
    dims: str | Sequence[int] | None = None,
    font_scale: float | None = None,
    render_plots: bool | None = None,
    opal_campaign: str | None = None,
    opal_run: str | None = None,
    opal_as_of_round: int | None = None,
    opal_fields: str | Sequence[str] | None = None,
    derive_ratio: Sequence[str] | None = None,
    attach_coords: bool = False,
    write: bool = False,
    allow_overwrite: bool = False,
    inplace: bool = False,
    out: str | Path | None = None,
    plot: Mapping[str, Any] | None = None,
    workspace_id: str | None = None,
    workspace_dir: str | Path | None = None,
) -> ClusterExecutionResult:
    root = _resolved_results_root(results_root)
    dims_value: str | None
    if dims is None:
        dims_value = None
    elif isinstance(dims, str):
        dims_value = dims
    else:
        dims_value = ",".join(str(value) for value in dims)
    try:
        execution = _run_umap(
            dataset=dataset,
            file=str(file) if file is not None else None,
            usr_root=str(usr_root) if usr_root is not None else None,
            name=name,
            key_col=key_col,
            x_col=x_col,
            x_cols=_normalized_csv(x_cols),
            neighbors=neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            preset=preset,
            color_by=_normalized_list(color_by) or ["cluster"],
            highlight=str(highlight) if highlight is not None else None,
            highlight_topn=highlight_topn,
            highlight_topn_col=highlight_topn_col,
            highlight_topn_asc=bool(highlight_topn_asc),
            highlight_hue_col=highlight_hue_col,
            alpha=alpha,
            size=size,
            dims=dims_value,
            font_scale=font_scale,
            render_plots=render_plots,
            opal_campaign=opal_campaign,
            opal_run=opal_run,
            opal_as_of_round=opal_as_of_round,
            opal_fields=_normalized_csv(opal_fields),
            derive_ratio=_normalized_list(derive_ratio),
            attach_coords=bool(attach_coords),
            write=bool(write),
            allow_overwrite=bool(allow_overwrite),
            inplace=bool(inplace),
            out=str(out) if out is not None else None,
            root=root,
            workspace_id=workspace_id,
            workspace_params={},
            workspace_plot=dict(plot or {}),
            console=None,
        )
    except Exception as exc:
        raise ClusterApiError(f"cluster umap failed: {exc}") from exc
    return ClusterExecutionResult(
        command="umap",
        subject=execution.subject,
        results_root=root,
        artifact_path=execution.artifact_path,
        workspace_id=workspace_id,
        workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
        run_record=_run_record_for_artifact(
            root,
            "umap",
            execution.artifact_path,
            alias=execution.run_record_subject,
        ),
    )


def run_analyze(
    *,
    results_root: str | Path,
    dataset: str | None = None,
    file: str | Path | None = None,
    usr_root: str | Path | None = None,
    cluster_col: str,
    group_by: str | Sequence[str] = "source",
    preset: str | None = None,
    out_dir: str | Path | None = None,
    composition: bool = False,
    diversity: bool = False,
    difffeat: bool = False,
    plots: bool = False,
    numeric: str | Sequence[str] | None = None,
    numeric_plots: bool = True,
    font_scale: float | None = None,
    opal_campaign: str | None = None,
    opal_as_of_round: int | None = None,
    opal_fields: str | Sequence[str] | None = None,
    plot: Mapping[str, Any] | None = None,
    workspace_id: str | None = None,
    workspace_dir: str | Path | None = None,
) -> ClusterExecutionResult:
    root = _resolved_results_root(results_root)
    try:
        execution = _run_analyze(
            dataset=dataset,
            file=str(file) if file is not None else None,
            usr_root=str(usr_root) if usr_root is not None else None,
            cluster_col=cluster_col,
            group_by=_normalized_csv(group_by) or "source",
            preset=preset,
            out_dir=str(out_dir) if out_dir is not None else None,
            composition=bool(composition),
            diversity=bool(diversity),
            difffeat=bool(difffeat),
            plots=bool(plots),
            numeric=_normalized_csv(numeric),
            numeric_plots=bool(numeric_plots),
            font_scale=font_scale,
            opal_campaign=opal_campaign,
            opal_as_of_round=opal_as_of_round,
            opal_fields=_normalized_csv(opal_fields),
            root=root,
            workspace_id=workspace_id,
            workspace_plot=dict(plot or {}),
            console=None,
        )
    except Exception as exc:
        raise ClusterApiError(f"cluster analyze failed: {exc}") from exc
    return ClusterExecutionResult(
        command="analyze",
        subject=execution.subject,
        results_root=root,
        artifact_path=execution.artifact_path,
        workspace_id=workspace_id,
        workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
        run_record=_run_record_for_artifact(
            root,
            "analyze",
            execution.artifact_path,
            alias=fit_alias_from_cluster_col(cluster_col) or execution.run_record_subject,
        ),
    )


def run_sweep(
    *,
    results_root: str | Path,
    dataset: str | None = None,
    file: str | Path | None = None,
    usr_root: str | Path | None = None,
    key_col: str = "id",
    x_col: str | None = None,
    x_cols: str | Sequence[str] | None = None,
    method: str = "leiden",
    preset: str | None = None,
    method_params: Mapping[str, Any] | None = None,
    res_min: float = 0.05,
    res_max: float = 1.0,
    step: float = 0.05,
    replicates: int = 5,
    seeds: str | Sequence[int] | None = None,
    out_dir: str | Path | None = None,
    workspace_id: str | None = None,
    workspace_dir: str | Path | None = None,
) -> ClusterExecutionResult:
    root = _resolved_results_root(results_root)
    seeds_value = _normalized_csv(seeds) if not isinstance(seeds, str) else seeds
    try:
        execution = _run_sweep(
            dataset=dataset,
            file=str(file) if file is not None else None,
            usr_root=str(usr_root) if usr_root is not None else None,
            key_col=key_col,
            x_col=x_col,
            x_cols=_normalized_csv(x_cols),
            method=method,
            preset=preset,
            method_params=dict(method_params or {}),
            res_min=float(res_min),
            res_max=float(res_max),
            step=float(step),
            replicates=int(replicates),
            seeds=seeds_value or "",
            out_dir=str(out_dir) if out_dir is not None else None,
            root=root,
            workspace_id=workspace_id,
            console=None,
        )
    except Exception as exc:
        raise ClusterApiError(f"cluster sweep failed: {exc}") from exc
    return ClusterExecutionResult(
        command="sweep",
        subject=execution.subject,
        results_root=root,
        artifact_path=execution.artifact_path,
        workspace_id=workspace_id,
        workspace_dir=Path(workspace_dir) if workspace_dir is not None else None,
        run_record=_run_record_for_artifact(
            root,
            "sweep",
            execution.artifact_path,
            alias=execution.run_record_subject,
        ),
    )


def run_fit_workspace(
    workspace: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ClusterExecutionResult:
    config = _workspace_context(workspace)
    params, _ = _merged_workspace_section(config, "fit", overrides)
    return run_fit(
        results_root=config.results_root,
        dataset=params.get("dataset"),
        file=params.get("file"),
        usr_root=params.get("usr_root"),
        name=params.get("name"),
        key_col=str(params.get("key_col", "id")),
        x_col=params.get("x_col"),
        x_cols=params.get("x_cols"),
        method=str(params.get("method", "leiden")),
        preset=params.get("preset"),
        method_params=dict(params.get("method_params", {}) or {}),
        silhouette=bool(params.get("silhouette", False)),
        full_silhouette=bool(params.get("full_silhouette", False)),
        dedupe_policy=str(params.get("dedupe_policy", "error")),
        reuse=str(params.get("reuse", "auto")),
        force=bool(params.get("force", False)),
        write=bool(params.get("write", False)),
        allow_overwrite=bool(params.get("allow_overwrite", False)),
        inplace=bool(params.get("inplace", False)),
        out=params.get("out"),
        workspace_id=config.workspace_id,
        workspace_dir=config.workspace_dir,
    )


def run_umap_workspace(
    workspace: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ClusterExecutionResult:
    config = _workspace_context(workspace)
    params, plot = _merged_workspace_section(config, "umap", overrides)
    return run_umap(
        results_root=config.results_root,
        dataset=params.get("dataset"),
        file=params.get("file"),
        usr_root=params.get("usr_root"),
        name=str(params.get("name") or config.workspace_id),
        key_col=str(params.get("key_col", "id")),
        x_col=params.get("x_col"),
        x_cols=params.get("x_cols"),
        neighbors=params.get("neighbors"),
        min_dist=params.get("min_dist"),
        metric=params.get("metric"),
        random_state=params.get("random_state"),
        preset=params.get("preset"),
        color_by=list(params.get("color_by", ["cluster"])),
        highlight=params.get("highlight"),
        highlight_topn=params.get("highlight_topn"),
        highlight_topn_col=params.get("highlight_topn_col"),
        highlight_topn_asc=bool(params.get("highlight_topn_asc", False)),
        highlight_hue_col=params.get("highlight_hue_col"),
        alpha=params.get("alpha"),
        size=params.get("size"),
        dims=params.get("dims"),
        font_scale=params.get("font_scale"),
        render_plots=None,
        opal_campaign=params.get("opal_campaign"),
        opal_run=params.get("opal_run"),
        opal_as_of_round=params.get("opal_as_of_round"),
        opal_fields=params.get("opal_fields"),
        derive_ratio=params.get("derive_ratio"),
        attach_coords=bool(params.get("attach_coords", False)),
        write=bool(params.get("write", False)),
        allow_overwrite=bool(params.get("allow_overwrite", False)),
        inplace=bool(params.get("inplace", False)),
        out=params.get("out"),
        plot=plot,
        workspace_id=config.workspace_id,
        workspace_dir=config.workspace_dir,
    )


def run_analyze_workspace(
    workspace: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ClusterExecutionResult:
    config = _workspace_context(workspace)
    params, plot = _merged_workspace_section(config, "analyze", overrides)
    return run_analyze(
        results_root=config.results_root,
        dataset=params.get("dataset"),
        file=params.get("file"),
        usr_root=params.get("usr_root"),
        cluster_col=str(params.get("cluster_col") or ""),
        group_by=params.get("group_by", "source"),
        preset=params.get("preset"),
        out_dir=params.get("out_dir"),
        composition=bool(params.get("composition", False)),
        diversity=bool(params.get("diversity", False)),
        difffeat=bool(params.get("difffeat", False)),
        plots=bool(params.get("plots", False)),
        numeric=params.get("numeric"),
        numeric_plots=bool(params.get("numeric_plots", True)),
        font_scale=params.get("font_scale"),
        opal_campaign=params.get("opal_campaign"),
        opal_as_of_round=params.get("opal_as_of_round"),
        opal_fields=params.get("opal_fields"),
        plot=plot,
        workspace_id=config.workspace_id,
        workspace_dir=config.workspace_dir,
    )


def run_sweep_workspace(
    workspace: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ClusterExecutionResult:
    config = _workspace_context(workspace)
    params, _ = _merged_workspace_section(config, "fit", overrides)
    result = run_sweep(
        results_root=config.results_root,
        dataset=params.get("dataset"),
        file=params.get("file"),
        usr_root=params.get("usr_root"),
        key_col=str(params.get("key_col", "id")),
        x_col=params.get("x_col"),
        x_cols=params.get("x_cols"),
        method=str(params.get("method", "leiden")),
        preset=params.get("preset"),
        method_params=dict(params.get("method_params", {}) or {}),
        res_min=float(params.get("res_min", 0.05)),
        res_max=float(params.get("res_max", 1.0)),
        step=float(params.get("step", 0.05)),
        replicates=int(params.get("replicates", 5)),
        seeds=params.get("seeds", "1,2,3,4,5"),
        out_dir=params.get("out_dir"),
        workspace_id=config.workspace_id,
        workspace_dir=config.workspace_dir,
    )
    if result.run_record is not None:
        return result
    return ClusterExecutionResult(
        command=result.command,
        subject=result.subject,
        results_root=result.results_root,
        artifact_path=result.artifact_path,
        workspace_id=result.workspace_id,
        workspace_dir=result.workspace_dir,
        run_record=_run_record_for_alias(config.results_root, "sweep", result.subject),
    )


__all__ = [
    "ClusterApiError",
    "ClusterExecutionResult",
    "list_runs",
    "list_workspace_runs",
    "run_analyze",
    "run_analyze_workspace",
    "run_fit",
    "run_fit_workspace",
    "run_sweep",
    "run_sweep_workspace",
    "run_umap",
    "run_umap_workspace",
]
