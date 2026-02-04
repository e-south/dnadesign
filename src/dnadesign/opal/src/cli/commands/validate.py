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
from ...storage.data_access import ESSENTIAL_COLS
from ..formatting import kv_block
from ..registry import cli_command
from ._common import (
    internal_error,
    load_cli_config,
    opal_error,
    resolve_config_path,
    store_from_cfg,
)


@cli_command("validate", help="End-to-end table checks (essentials present; X column present).")
def cmd_validate(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()

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
                "Table shape": f"{df.shape[0]} rows × {df.shape[1]} cols",
            },
        )
        print_stdout(ctx)

        missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
        if missing:
            raise OpalError(f"Missing essential columns: {missing}")
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
        if cfg.data.x_column_name not in df.columns:
            # Helpful hints: case-only match and fuzzy suggestions
            target = cfg.data.x_column_name
            cols = list(map(str, df.columns))
            case_only = [c for c in cols if c.lower() == target.lower()]
            fuzzy = get_close_matches(target, cols, n=5, cutoff=0.6)
            hint_lines = []
            if case_only and target not in case_only:
                hint_lines.append(f"Case-only match found: {case_only[0]!r}")
            if fuzzy:
                hint_lines.append("Similar columns: " + ", ".join(repr(c) for c in fuzzy))
            hint = (" " + " | ".join(hint_lines)) if hint_lines else ""
            raise OpalError(f"Missing X column: {target}.{hint}")
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

    except OpalError as e:
        opal_error("validate", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("validate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
