"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution_fit.py

Fit execution runtime for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from .execution_fit_support import (
    apply_fit_attachment,
    build_fit_attach_columns,
    build_reused_fit_attach_columns,
    fit_meta_json,
    load_fit_input,
)
from .execution_support import (
    CommandExecution,
    _apply_dedupe,
    _collect_existing_meta_sig,
    _log,
    _rows_ids,
    _rule,
    append_command_record_or_warn,
    cluster_overlay_col,
    print_fit_summary,
    progress_scope,
)
from .io.read import extract_X
from .methods.registry import get_method
from .presets.runtime import apply_preset
from .runs.contracts import ClusterRun, RunCounts, utc_now_iso
from .runs.recorder import CommandRecord, record_fit_run
from .runs.reuse import find_equivalent_fit
from .runs.signatures import InputSignature, MethodSignature, file_fingerprint, ids_hash
from .runtime_contracts import FeatureSpec, FitRequest, InputSource, MethodConfig
from .util.checks import ClusterError, assert_id_sequence_bijection
from .util.slug import artifact_slug, auto_run_name, slugify


def run_fit(
    *,
    dataset: str | None,
    file: str | None,
    usr_root: str | None,
    name: str | None,
    key_col: str,
    x_col: str | None,
    x_cols: str | None,
    method: str,
    preset: str | None,
    method_params: dict[str, Any],
    silhouette: bool,
    full_silhouette: bool,
    dedupe_policy: str,
    reuse: str,
    force: bool,
    write: bool,
    allow_overwrite: bool,
    inplace: bool,
    out: str | None,
    root: Path,
    workspace_id: str | None = None,
    console: Console | None = None,
) -> CommandExecution:
    if name:
        name = slugify(name)
    _rule(console, "[bold]cluster fit[/]")
    feature_spec = FeatureSpec.from_inputs(x_col=x_col, x_cols=x_cols)
    loaded_input = load_fit_input(
        dataset=dataset,
        file=file,
        usr_root=usr_root,
        key_col=key_col,
        feature_spec=feature_spec,
        write=write,
    )
    ictx = loaded_input.context
    df = loaded_input.fit_df
    attach_base_df = loaded_input.attach_base_df
    _log(console, "log", f"Input: kind={ictx['kind']} ref={ictx.get('dataset') or ictx.get('file')}")
    df = _apply_dedupe(df, key_col=key_col, policy=dedupe_policy)
    try:
        assert_id_sequence_bijection(df, id_col=key_col, seq_col="sequence")
    except ClusterError as exc:
        raise typer.BadParameter(str(exc)) from exc

    with progress_scope(console) as progress:
        task = progress.add_task("Preparing X...", total=None)
        X = extract_X(
            df,
            x_col=feature_spec.columns[0] if feature_spec.mode == "single_col" else None,
            x_cols=list(feature_spec.columns) if feature_spec.mode == "multi_col" else None,
        )
        ids = _rows_ids(df, key_col)
        progress.update(task, completed=1)

    method_spec = get_method(method)
    preset_params = apply_preset("method", preset)
    resolved_method_params = method_spec.resolve_fit_params(preset=preset_params, raw_params=method_params)
    source = InputSource.from_context(ictx)
    method_config = MethodConfig(method_id=method_spec.method_id, params=resolved_method_params)
    fit_request = FitRequest(source=source, key_col=key_col, feature=feature_spec, method=method_config)

    input_sig = InputSignature(
        **fit_request.input_signature_payload(
            row_ids_hash=ids_hash(ids),
            x_dim=int(X.shape[1]),
            fingerprint=file_fingerprint(source.file),
        )
    )
    input_hash = input_sig.hash()
    method_signature = MethodSignature(method_id=method_spec.method_id, params=resolved_method_params, libs={})
    method_sig = method_signature.hash()

    if not force and reuse in ("auto", "require", "reattach"):
        hit = find_equivalent_fit(input_hash, method_sig, root=root)
        if hit is not None:
            existing_sig = _collect_existing_meta_sig(df, name or hit.get("alias") or hit.get("run_slug"))
            if reuse in ("auto", "require") and existing_sig == method_sig:
                _log(console, "print", "[green]Reuse[/green]: matching fit already attached; nothing to do.")
                return CommandExecution(
                    command="fit",
                    subject=name or hit.get("alias") or hit.get("run_slug") or "fit",
                    artifact_path=Path(hit["labels_path"]).parent,
                    run_record_subject=name or hit.get("alias"),
                )
            if reuse in ("auto", "reattach") and write:
                try:
                    run_alias = name or str(hit["alias"])
                    attach_cols = build_reused_fit_attach_columns(
                        df=df,
                        key_col=key_col,
                        run_alias=run_alias,
                        labels_path=hit["labels_path"],
                        meta_json=fit_meta_json(
                            method_id=method_spec.method_id,
                            feature_label=feature_spec.primary_label,
                            n_rows=len(df),
                            method_params=resolved_method_params,
                            source_clause=source.source_clause(),
                            method_sig=method_sig,
                        ),
                    )
                    apply_fit_attachment(
                        ctx=ictx,
                        attach_cols=attach_cols,
                        key_col=key_col,
                        allow_overwrite=allow_overwrite,
                        inplace=inplace,
                        out=out,
                        attach_base_df=attach_base_df,
                        console=console,
                    )
                    _log(
                        console,
                        "print",
                        "[green]Reattached[/green] labels from cache to "
                        + ("USR dataset." if ictx["kind"] == "usr" else "file."),
                    )
                    return CommandExecution(
                        command="fit",
                        subject=run_alias,
                        artifact_path=Path(hit["labels_path"]).parent,
                        run_record_subject=run_alias,
                    )
                except Exception as exc:
                    if reuse == "require":
                        raise RuntimeError(f"Reuse required but reattach failed: {exc}") from exc

    with progress_scope(console) as progress:
        task = progress.add_task(f"Clustering ({method_spec.display_name})...", total=None)
        labels = method_spec.fit(X, **resolved_method_params)
        progress.update(task, completed=1)

    quality = None
    if silhouette:
        try:
            from sklearn.metrics import silhouette_samples
        except Exception:
            _log(console, "print", "[yellow]Silhouette requested but scikit-learn is missing. Skipping.[/yellow]")
        else:
            with progress_scope(console) as progress:
                task = progress.add_task("Computing silhouette...", total=None)
                row_count = len(df)
                sil_metric = str(resolved_method_params.get("metric", "euclidean"))
                sil_seed = int(resolved_method_params.get("random_state", 42))
                if row_count > 20000 and not full_silhouette:
                    rng = np.random.default_rng(sil_seed)
                    keep = rng.choice(np.arange(row_count), size=20000, replace=False)
                    svals = np.full(row_count, np.nan, dtype="float32")
                    svals[keep] = silhouette_samples(X[keep], labels[keep], metric=sil_metric).astype("float32")
                    quality = svals
                else:
                    quality = silhouette_samples(X, labels, metric=sil_metric).astype("float32")
                progress.update(task, completed=1)

    run_alias = name or auto_run_name(method_spec.default_run_prefix, method_spec.slug_params(resolved_method_params))
    created_utc = utc_now_iso()
    run_slug = artifact_slug(
        run_alias,
        created_utc=created_utc,
        fingerprint=f"{input_hash}:{method_sig}",
    )
    attached_columns = [cluster_overlay_col(run_alias), cluster_overlay_col(run_alias, "meta")]
    meta_json = fit_meta_json(
        method_id=method_spec.method_id,
        feature_label=feature_spec.primary_label,
        n_rows=len(df),
        method_params=resolved_method_params,
        source_clause=source.source_clause(),
        method_sig=method_sig,
    )
    attach_cols = build_fit_attach_columns(
        df=df,
        key_col=key_col,
        run_alias=run_alias,
        labels=labels,
        meta_json=meta_json,
        quality=quality,
    )
    if quality is not None:
        attached_columns.append(cluster_overlay_col(run_alias, "quality"))

    n_clusters = int(len(np.unique(labels)))
    cluster_run = ClusterRun(
        alias=run_alias,
        slug=run_slug,
        created_utc=created_utc,
        input_signature=input_sig,
        method_signature=method_signature,
        source=source,
        feature=feature_spec,
        x_dim=int(X.shape[1]),
        counts=RunCounts(n_rows=int(len(df)), n_clusters=n_clusters),
        wrote_usr_columns=bool(ictx["kind"] == "usr"),
        attached_columns=tuple(attached_columns),
    )
    size_counts = pd.Series(labels).value_counts().to_dict()
    run_dir = record_fit_run(
        root=root,
        run=cluster_run,
        labels_df=pd.DataFrame({"id": df[key_col].astype(str), "cluster_label": labels}),
        summary={"cluster_sizes": size_counts},
        input_sig_hash=input_hash,
        method_sig_hash=method_sig,
    )

    if not write:
        _log(
            console,
            "print",
            "[yellow]Dry-run[/yellow]: computed labels but did not write to the table. Use --write to apply.",
        )
        _log(console, "print", f"Run recorded under [bold]{run_dir}[/].")
        print_fit_summary(labels, run_alias, size_counts, console=console)
        return CommandExecution(
            command="fit",
            subject=run_alias,
            artifact_path=run_dir,
            run_record_subject=run_alias,
        )

    try:
        apply_fit_attachment(
            ctx=ictx,
            attach_cols=attach_cols,
            key_col=key_col,
            allow_overwrite=allow_overwrite,
            inplace=inplace,
            out=out,
            attach_base_df=attach_base_df,
            console=console,
        )
    except Exception as exc:
        if "Columns already exist" in str(exc) and not allow_overwrite:
            raise RuntimeError(
                "Columns already exist. Re-run with `-y/--allow-overwrite` or choose a new --name."
            ) from exc
        raise

    print_fit_summary(labels, run_alias, size_counts, console=console)
    append_command_record_or_warn(
        root / run_alias,
        CommandRecord(
            command="fit",
            subject=run_alias,
            workspace=workspace_id,
            preset=preset or None,
            resolved={"name": run_alias, "method": method_spec.method_id, **resolved_method_params},
        ),
        console=console,
    )
    return CommandExecution(command="fit", subject=run_alias, artifact_path=run_dir, run_record_subject=run_alias)


__all__ = ["run_fit"]
