"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/validate.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json as _json
from difflib import get_close_matches
from pathlib import Path

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...registries.models import get_model
from ...registries.objectives import get_objective_declared_channels
from ...storage.data_access import ESSENTIAL_COLS
from ...storage.label_sources import label_source_status
from ...storage.x_contracts import validate_x_parquet_column
from ..formatting import kv_block
from ..guidance_hints import maybe_print_hints
from ..registry import cli_command
from ._common import (
    internal_error,
    load_cli_config,
    opal_error,
    resolve_config_path,
    store_from_cfg,
)


def _parse_channel_ref(ref: object, *, label: str) -> tuple[str, str]:
    key = str(ref or "").strip()
    if not key:
        raise OpalError(f"{label} is required and must be a non-empty channel reference.")
    if "/" not in key:
        raise OpalError(f"{label} must be in '<objective>/<channel>' format; got {key!r}.")
    objective_name, channel_name = key.split("/", 1)
    objective_name = objective_name.strip()
    channel_name = channel_name.strip()
    if not objective_name or not channel_name:
        raise OpalError(f"{label} must be in '<objective>/<channel>' format; got {key!r}.")
    return objective_name, channel_name


def _validate_selection_channel_refs(cfg) -> None:
    sel_name = str(cfg.selection.selection.name)
    sel_params = dict(cfg.selection.selection.params or {})
    objective_names = {str(o.name) for o in cfg.objectives.objectives}

    score_obj, score_channel = _parse_channel_ref(sel_params.get("score_ref"), label="selection.params.score_ref")
    if score_obj not in objective_names:
        raise OpalError(
            "selection.params.score_ref references objective "
            f"{score_obj!r}, which is not configured in objectives={sorted(objective_names)}."
        )
    score_declared = get_objective_declared_channels(score_obj)
    score_decl = score_declared.get("score", ())
    if score_decl and score_channel not in set(score_decl):
        available = [f"{score_obj}/{ch}" for ch in score_decl]
        raise OpalError(
            f"score_ref channel '{score_obj}/{score_channel}' not declared by objective '{score_obj}'. "
            f"Available: {available}"
        )
    selected_mode = str(sel_params.get("objective_mode", "")).strip().lower()
    score_modes = dict(score_declared.get("score_modes", {}) or {})
    if score_modes and score_channel in score_modes:
        channel_mode = str(score_modes[score_channel]).strip().lower()
        if selected_mode and channel_mode != selected_mode:
            raise OpalError(
                "Objective mode mismatch: selected score channel "
                f"'{score_obj}/{score_channel}' has mode '{channel_mode}' but "
                f"selection objective_mode is '{selected_mode}'."
            )

    uncertainty_ref_raw = sel_params.get("uncertainty_ref")
    if uncertainty_ref_raw is not None and str(uncertainty_ref_raw).strip():
        unc_obj, unc_channel = _parse_channel_ref(
            uncertainty_ref_raw,
            label="selection.params.uncertainty_ref",
        )
        if unc_obj not in objective_names:
            raise OpalError(
                "selection.params.uncertainty_ref references objective "
                f"{unc_obj!r}, which is not configured in objectives={sorted(objective_names)}."
            )
        unc_declared = get_objective_declared_channels(unc_obj)
        unc_decl = unc_declared.get("uncertainty", ())
        if unc_decl and unc_channel not in set(unc_decl):
            available = [f"{unc_obj}/{ch}" for ch in unc_decl]
            raise OpalError(
                f"uncertainty_ref channel '{unc_obj}/{unc_channel}' not declared by objective '{unc_obj}'. "
                f"Available: {available}"
            )

    if sel_name == "expected_improvement":
        if uncertainty_ref_raw is None or not str(uncertainty_ref_raw).strip():
            raise OpalError("selection.params.uncertainty_ref is required for expected_improvement.")
        model = get_model(cfg.model.name, cfg.model.params)
        contract = getattr(model, "__opal_contract__", None)
        produces_by_stage = getattr(contract, "produces_by_stage", None) or {}
        predict_produces = tuple(produces_by_stage.get("predict", ()) or ())
        if "model/<self>/std_devs" not in predict_produces:
            raise OpalError(
                "selection='expected_improvement' requires a model plugin that emits predictive std "
                "('model/<self>/std_devs'). "
                f"Configured model '{cfg.model.name}' does not advertise that contract."
            )


@cli_command("validate", help="End-to-end table checks (essentials present; X column finite and fixed-length).")
def cmd_validate(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable next-step hints in text output."),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        schema_columns = store.schema_columns()
        row_count = store.row_count()

        # Always print absolute context so there's no ambiguity
        cfg_abs = str(Path(cfg_path).resolve())
        wd_abs = str(Path(cfg.campaign.workdir).resolve())
        rec_abs = str(store.records_path.resolve())
        ctx = kv_block(
            "Context",
            {
                "Config": cfg_abs,
                "Workdir": wd_abs,
                "Records": rec_abs,
                "X (YAML)": cfg.data.x_column_name,
                "Y (YAML)": cfg.data.y_column_name,
                "Table shape": f"{row_count} rows × {len(schema_columns)} cols",
            },
        )
        print_stdout(ctx)

        missing = [c for c in ESSENTIAL_COLS if c not in schema_columns]
        if missing:
            raise OpalError(f"Missing essential columns: {missing}")

        label_hist_col = store.label_hist_col()
        scalar_columns = [*ESSENTIAL_COLS]
        if cfg.data.y_column_name in schema_columns:
            scalar_columns.append(cfg.data.y_column_name)
        if label_hist_col in schema_columns:
            scalar_columns.append(label_hist_col)
        df = store.load_columns(scalar_columns)

        # Enforce unique ids
        ids = df["id"].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].unique().tolist()[:10]
            raise OpalError(f"Duplicate ids detected in records.parquet (sample={dup}).")

        if cfg.safety.require_biotype_and_alphabet_on_init:
            for col in ("bio_type", "alphabet"):
                if df[col].isna().any():
                    bad = df.loc[df[col].isna(), "id"].astype(str).tolist()[:10]
                    raise OpalError(f"Missing values in '{col}' (sample ids={bad}).")
        if cfg.data.x_column_name not in schema_columns:
            # Helpful hints: case-only match and fuzzy suggestions
            target = cfg.data.x_column_name
            cols = list(map(str, schema_columns))
            case_only = [c for c in cols if c.lower() == target.lower()]
            fuzzy = get_close_matches(target, cols, n=5, cutoff=0.6)
            hint_lines = []
            if case_only and target not in case_only:
                hint_lines.append(f"Case-only match found: {case_only[0]!r}")
            if fuzzy:
                hint_lines.append("Similar columns: " + ", ".join(repr(c) for c in fuzzy))
            hint = (" " + " | ".join(hint_lines)) if hint_lines else ""
            raise OpalError(f"Missing X column: {target}.{hint}")
        x_report = validate_x_parquet_column(
            store.records_path,
            x_column=cfg.data.x_column_name,
            id_column="id",
        )
        print_stdout(
            kv_block(
                "X contract",
                {
                    "Rows": x_report.row_count,
                    "X dim": x_report.x_dim,
                },
            )
        )
        # If Y exists, enforce vector length (sampled) and that labeled rows have X
        ycol = cfg.data.y_column_name
        if ycol in df.columns:
            exp_len = cfg.data.y_expected_length
            if exp_len:
                import numpy as _np
                import pandas as _pd

                def _parse_vec(v):
                    """
                    Return (ok:bool, vec:list[float] | None, reason:str). ok=False
                    if any element is NaN/Inf or length != exp_len. Whole-cell None/NaN
                    means 'no label' and is skipped by caller (we dropna() before calling).
                    """
                    # containers
                    if isinstance(v, (list, tuple, _np.ndarray, _pd.Series)):
                        arr = _np.asarray(v, dtype=float).ravel()
                        if arr.size != exp_len:
                            return (
                                False,
                                list(map(float, arr)) if arr.size else [],
                                "wrong_length",
                            )
                        if not _np.all(_np.isfinite(arr)):
                            return False, list(map(float, arr)), "non_finite"
                        return True, list(map(float, arr)), ""
                    # strings (expect JSON-like "[..]"; reject if contains 'nan'/'inf' tokens)
                    if isinstance(v, str):
                        s = v.strip()
                        if s.startswith("[") and s.endswith("]"):
                            lower = s.lower()
                            if "nan" in lower or "inf" in lower:
                                return False, None, "non_finite_token"
                            inner = s[1:-1].strip()
                            parts = [] if inner == "" else [p.strip() for p in inner.split(",")]
                            try:
                                arr = _np.asarray([float(p) for p in parts], dtype=float)
                            except Exception:
                                return False, None, "non_numeric"
                            if arr.size != exp_len:
                                return (
                                    False,
                                    list(map(float, arr)) if arr.size else [],
                                    "wrong_length",
                                )
                            if not _np.all(_np.isfinite(arr)):
                                return False, list(map(float, arr)), "non_finite"
                            return True, list(map(float, arr)), ""
                        # scalar-like string → treat as scalar (invalid)
                        return False, None, "scalar_string"
                    # other scalar types
                    try:
                        _ = float(v)
                        return False, None, "scalar_value"
                    except Exception:
                        return False, None, "unknown_type"

                # Check all non-null labeled rows (entire cell None/NaN is allowed = unlabeled)
                sample_series = df[ycol].dropna()
                bad_examples = []
                bad_count = 0
                for idx, val in sample_series.items():
                    ok, vec, reason = _parse_vec(val)
                    if not ok:
                        bad_count += 1
                        if len(bad_examples) < 3:
                            bad_examples.append(
                                {
                                    "id": (str(df.at[idx, "id"]) if "id" in df.columns else f"row_{idx}"),
                                    "reason": reason,
                                    "value_preview": (str(val)[:120] + ("…" if len(str(val)) > 120 else "")),
                                }
                            )
                if bad_count:
                    raise OpalError(
                        f"Y validation failed: require length={exp_len} with all finite numbers per labeled row; "
                        f"violations={bad_count}/{len(sample_series)}. Examples: {_json.dumps(bad_examples, ensure_ascii=False)}"  # noqa
                    )
        # Validate label_hist when present (strict)
        try:
            store.validate_label_hist(df, require=False)
        except OpalError as e:
            raise OpalError(f"label_hist validation failed: {e}")

        labels_status = label_source_status(cfg, store, df, strict=False)
        if labels_status.get("exists") and labels_status.get("valid") is False:
            raise OpalError(f"label source validation failed: {labels_status.get('error')}")
        label_rows = {
            "kind": labels_status.get("kind"),
            "exists": labels_status.get("exists"),
            "valid": labels_status.get("valid"),
            "label_count": labels_status.get("label_count"),
            "available_rounds": ",".join(map(str, labels_status.get("available_rounds") or [])) or "(none)",
            "prediction_records": cfg.writeback.prediction_records,
        }
        if labels_status.get("kind") == "usr_sidecar":
            label_rows["path"] = labels_status.get("path")
            label_rows["y_space"] = labels_status.get("y_space")
        else:
            label_rows["column"] = labels_status.get("column")
        print_stdout(kv_block("Label source", label_rows))

        _validate_selection_channel_refs(cfg)

        # Emit a single, clear success line (with a note if CWD is outside workdir).
        msg = "OK: validation passed."
        try:
            cwd = Path.cwd().resolve()
            wd = Path(cfg.campaign.workdir).resolve()
            if wd not in cwd.parents and cwd != wd:
                msg = f"{msg} (Note: your CWD '{cwd}' is outside campaign workdir '{wd}')"
        except Exception:
            pass
        print_stdout(msg)
        maybe_print_hints(command_name="validate", cfg_path=cfg_path, no_hints=no_hints, json_output=False)

    except OpalError as e:
        opal_error("validate", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("validate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
