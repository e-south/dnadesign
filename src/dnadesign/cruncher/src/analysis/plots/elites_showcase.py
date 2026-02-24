"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/elites_showcase.py

Render elite-focused motif placement showcase panels via the public baserender API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.baserender import (
    Display,
    Effect,
    Feature,
    Record,
    Span,
    cruncher_showcase_style_overrides,
    render_record_grid_figure,
)
from dnadesign.cruncher.analysis.plots._savefig import savefig

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_OVERLAY_MAX_CHARS = 80
_NORM_SCORE_EPS = 1.0e-9


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _required_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _ordered_elites(elites_df: pd.DataFrame) -> pd.DataFrame:
    frame = elites_df.copy()
    frame["id"] = frame["id"].astype(str)
    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.sort_values(["rank", "id"], na_position="last")
    else:
        frame = frame.sort_values("id")
    return frame


def _hits_index(hits_df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    _required_columns(hits_df, ["elite_id", "tf", "best_start", "pwm_width", "best_strand"], context="hits_df")
    key_counts = hits_df.groupby(["elite_id", "tf"]).size()
    duplicates = key_counts[key_counts > 1]
    if not duplicates.empty:
        labels = [f"({idx[0]}, {idx[1]}) x{int(count)}" for idx, count in duplicates.items()]
        raise ValueError(f"hits_df contains duplicate elite/tf rows: {labels}")

    rows: dict[tuple[str, str], pd.Series] = {}
    for _, row in hits_df.iterrows():
        rows[(str(row["elite_id"]), str(row["tf"]))] = row
    return rows


def _normalized_score_from_row(
    elite_row: pd.Series,
    *,
    tf_name: str,
) -> float:
    norm_column = f"norm_{tf_name}"
    if norm_column not in elite_row.index:
        raise ValueError(f"elites_df missing required columns: ['{norm_column}']")
    value = pd.to_numeric(elite_row.get(norm_column), errors="coerce")
    if pd.isna(value):
        raise ValueError(f"elites_df column '{norm_column}' must be numeric for showcase overlay titles.")
    numeric = float(value)
    if numeric < -_NORM_SCORE_EPS or numeric > 1.0 + _NORM_SCORE_EPS:
        raise ValueError(f"elites_df column '{norm_column}' must be normalized in [0,1] for showcase overlay titles.")
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _validate_normalized_columns(elites_df: pd.DataFrame, *, tf_names: list[str]) -> None:
    for tf_name in tf_names:
        norm_column = f"norm_{tf_name}"
        if norm_column not in elites_df.columns:
            raise ValueError(f"elites_df missing required columns: ['{norm_column}']")
        numeric = pd.to_numeric(elites_df[norm_column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"elites_df column '{norm_column}' must be numeric for elites_showcase.")
        below = numeric < -_NORM_SCORE_EPS
        above = numeric > 1.0 + _NORM_SCORE_EPS
        if bool(below.any()) or bool(above.any()):
            raise ValueError(f"elites_df column '{norm_column}' must be normalized in [0,1] for elites_showcase.")
        elites_df[norm_column] = numeric.astype(float).clip(lower=0.0, upper=1.0)


def _overlay_text(
    elite_row: pd.Series,
    *,
    tf_names: Iterable[str] | None = None,
    max_chars: int = _OVERLAY_MAX_CHARS,
) -> str:
    if max_chars < 6:
        raise ValueError("elites showcase title max_chars must be >= 6")
    rank_value = elite_row.get("rank")
    subject = "Elite"
    if rank_value is not None and pd.notna(rank_value):
        rank_numeric = pd.to_numeric(rank_value, errors="coerce")
        if pd.notna(rank_numeric):
            subject = f"Elite #{int(rank_numeric)}"
    hash_token: str | None = None
    hash_candidate = elite_row.get("hash_id")
    if isinstance(hash_candidate, str) and hash_candidate.strip():
        hash_token = hash_candidate.strip()
    elif "sequence" in elite_row.index:
        sequence = str(elite_row.get("sequence") or "").strip().upper()
        elite_id = str(elite_row.get("id") or "").strip()
        if sequence and elite_id:
            hash_token = hashlib.sha256(f"{elite_id}|{sequence}".encode("utf-8")).hexdigest()[:12]
    if hash_token:
        subject = f"{subject} [{hash_token}]"
    tf_list = [str(tf).strip() for tf in (tf_names or []) if str(tf).strip()]
    if not tf_list:
        return subject[:max_chars].rstrip()
    score_tokens: list[str] = []
    for tf_name in tf_list:
        score_value = _normalized_score_from_row(elite_row, tf_name=tf_name)
        score_tokens.append(f"{tf_name}={score_value:.2f}")
    title = f"{subject}\n{' '.join(score_tokens)}"
    return title[:max_chars].rstrip()


_SHOWCASE_STYLE_OVERRIDES: Mapping[str, object] = cruncher_showcase_style_overrides()
OverlayTextFn = Callable[[pd.Series, list[str]], str]


def _matrix_from_pwm(pwm_obj) -> list[list[float]]:
    matrix = getattr(pwm_obj, "matrix", None)
    if matrix is None:
        raise ValueError("PWM object missing matrix")
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("PWM matrix must be 2D with at least 4 columns [A,C,G,T]")
    return [[float(v) for v in row[:4]] for row in arr.tolist()]


def build_elites_showcase_records(
    *,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
    pwms: Mapping[str, object],
    max_panels: int,
    overlay_text_fn: OverlayTextFn | None = None,
    meta_source: str = "cruncher_elites_showcase",
) -> list[Record]:
    if elites_df is None or elites_df.empty:
        raise ValueError("Elites table is required for elites_showcase.")
    if hits_df is None or hits_df.empty:
        raise ValueError("Hits table is required for elites_showcase.")
    if not isinstance(max_panels, int) or max_panels < 1:
        raise ValueError("analysis.elites_showcase.max_panels must be >= 1")

    tf_list = [str(tf) for tf in tf_names]
    if not tf_list:
        raise ValueError("TF names are required for elites_showcase.")
    required_elite_columns = ["id", "sequence", *[f"norm_{tf_name}" for tf_name in tf_list]]
    _required_columns(elites_df, required_elite_columns, context="elites_df")
    _validate_normalized_columns(elites_df, tf_names=tf_list)
    missing_tf_pwms = sorted(tf for tf in tf_list if tf not in pwms)
    if missing_tf_pwms:
        raise ValueError(f"Missing PWM(s) for elites_showcase TFs: {missing_tf_pwms}")
    pwm_matrix_by_tf = {tf_name: _matrix_from_pwm(pwms[tf_name]) for tf_name in tf_list}

    ordered = _ordered_elites(elites_df)
    if len(ordered) > max_panels:
        raise ValueError(
            f"analysis.elites_showcase.max_panels={max_panels} but elites_count={len(ordered)}; "
            "reduce elites or raise max_panels."
        )

    hit_by_key = _hits_index(hits_df)
    overlay_builder = (
        overlay_text_fn
        if overlay_text_fn is not None
        else (lambda elite_row, tf_list_: _overlay_text(elite_row, tf_names=tf_list_))
    )
    records: list[Record] = []
    for _, elite in ordered.iterrows():
        elite_id = str(elite["id"])
        sequence = str(elite["sequence"]).upper()
        features: list[Feature] = []
        effects: list[Effect] = []
        tag_labels: dict[str, str] = {}

        for tf_idx, tf_name in enumerate(tf_list):
            hit = hit_by_key.get((elite_id, tf_name))
            if hit is None:
                raise ValueError(f"Missing hit row for elite '{elite_id}' and TF '{tf_name}'")

            start = int(hit["best_start"])
            width = int(hit["pwm_width"])
            if width < 1:
                raise ValueError(f"Invalid pwm_width for elite '{elite_id}' and TF '{tf_name}': {width}")
            end = start + width
            if start < 0 or end > len(sequence):
                raise ValueError(
                    f"Hit span out of bounds for elite '{elite_id}' and TF '{tf_name}': "
                    f"[{start}, {end}) for sequence length {len(sequence)}"
                )

            strand_raw = str(hit["best_strand"]).strip()
            if strand_raw == "+":
                strand = "fwd"
            elif strand_raw == "-":
                strand = "rev"
            else:
                raise ValueError(
                    "Invalid best_strand for elite "
                    f"'{elite_id}' and TF '{tf_name}': {strand_raw!r}; "
                    "expected '+' or '-'"
                )

            segment = sequence[start:end]
            label = segment if strand == "fwd" else _revcomp(segment)
            tag = f"tf:{tf_name}"
            feature_id = f"{elite_id}:best_window:{tf_name}:{tf_idx}"
            matrix = pwm_matrix_by_tf[tf_name]
            if len(matrix) != width:
                raise ValueError(
                    f"PWM length mismatch for TF '{tf_name}': matrix rows={len(matrix)} "
                    f"but hit pwm_width={width} for elite '{elite_id}'"
                )

            features.append(
                Feature(
                    id=feature_id,
                    kind="kmer",
                    span=Span(start=start, end=end, strand=strand),
                    label=label,
                    tags=(tag,),
                    attrs={"tf": tf_name},
                    render={"priority": 10},
                )
            )
            effects.append(
                Effect(
                    kind="motif_logo",
                    target={"feature_id": feature_id},
                    params={"matrix": matrix},
                    render={"priority": 20},
                )
            )
            tag_labels[tag] = tf_name

        records.append(
            Record(
                id=elite_id,
                alphabet="DNA",
                sequence=sequence,
                features=tuple(features),
                effects=tuple(effects),
                display=Display(
                    overlay_text=overlay_builder(elite, tf_list),
                    tag_labels=tag_labels,
                ),
                meta={"source": meta_source},
            )
        )
    return records


def plot_elites_showcase(
    *,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: Iterable[str],
    pwms: Mapping[str, object],
    out_path: Path,
    max_panels: int,
    dpi: int,
    png_compress_level: int,
) -> None:
    records = build_elites_showcase_records(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=tf_names,
        pwms=pwms,
        max_panels=max_panels,
    )
    ncols = len(records)
    fig = render_record_grid_figure(
        records,
        ncols=ncols,
        style_overrides=_SHOWCASE_STYLE_OVERRIDES,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_path, dpi=dpi, png_compress_level=png_compress_level)
    plt.close(fig)
