"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/fimo_concordance.py

Compute descriptive optimizer-vs-FIMO concordance metrics from saved run artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.integrations.meme_suite import resolve_executable

_HEADER_RE = re.compile(r"[\s\-]+")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _normalize_header(name: str) -> str:
    return _HEADER_RE.sub("_", str(name).strip().lower())


def _sanitize_id(text: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", str(text).strip())
    return cleaned or "motif"


def _parse_fimo_tsv(text: str) -> list[dict[str, object]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return []
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader, None)
    if header is None:
        return []
    alias = {"pvalue": "p_value", "qvalue": "q_value", "sequence": "sequence_name"}
    normalized = [alias.get(_normalize_header(col), _normalize_header(col)) for col in header]
    idx = {name: i for i, name in enumerate(normalized)}
    required = {"sequence_name", "start", "stop", "strand", "score", "p_value"}
    missing = sorted(required - set(idx))
    if missing:
        raise ValueError(f"FIMO output missing required columns: {missing}")

    parsed: list[dict[str, object]] = []
    for row in reader:
        if not row:
            continue
        parsed.append(
            {
                "sequence_name": str(row[idx["sequence_name"]]),
                "start": int(row[idx["start"]]),
                "stop": int(row[idx["stop"]]),
                "strand": str(row[idx["strand"]]),
                "score": float(row[idx["score"]]),
                "p_value": float(row[idx["p_value"]]),
            }
        )
    return parsed


def _run_fimo(
    *,
    motif_path: Path,
    fasta_path: Path,
    bidirectional: bool,
    threshold: float,
    tool_path: Path | None,
) -> list[dict[str, object]]:
    exe = resolve_executable("fimo", tool_path=tool_path)
    if exe is None:
        raise FileNotFoundError(
            "FIMO executable not found. Install MEME Suite and ensure `fimo` is available on PATH or via MEME_BIN."
        )
    cmd = [str(exe), "--text", "--thresh", str(float(threshold))]
    if not bidirectional:
        cmd.append("--norc")
    cmd.extend([str(motif_path), str(fasta_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"FIMO failed (exit {result.returncode}). {stderr or 'No stderr output.'}")
    return _parse_fimo_tsv(result.stdout)


def _write_minimal_meme_motif(pwm: PWM, out_path: Path) -> str:
    motif_id = _sanitize_id(pwm.name)
    lines = [
        "MEME version 4",
        "",
        "ALPHABET= ACGT",
        "",
        "strands: + -",
        "",
        "Background letter frequencies:",
        "A 0.25 C 0.25 G 0.25 T 0.25",
        "",
        f"MOTIF {motif_id}",
        f"letter-probability matrix: alength= 4 w= {int(pwm.length)}",
    ]
    for row in np.asarray(pwm.matrix, dtype=float):
        lines.append(f"{float(row[0]):.6g} {float(row[1]):.6g} {float(row[2]):.6g} {float(row[3]):.6g}")
    out_path.write_text("\n".join(lines) + "\n")
    return motif_id


def _write_candidates_fasta(records: list[tuple[str, str]], out_path: Path) -> None:
    lines: list[str] = []
    for rec_id, seq in records:
        lines.append(f">{rec_id}")
        lines.append(str(seq))
    out_path.write_text("\n".join(lines) + "\n")


def _best_p_values_by_sequence(rows: list[dict[str, object]]) -> dict[str, float]:
    best: dict[str, tuple[float, int, int]] = {}
    for row in rows:
        seq_name = str(row["sequence_name"])
        p_value = float(row["p_value"])
        start = int(row["start"])
        stop = int(row["stop"])
        prev = best.get(seq_name)
        cur = (p_value, start, stop)
        if prev is None or cur < prev:
            best[seq_name] = cur
    return {name: value[0] for name, value in best.items()}


def _sequence_level_fimo_score(*, best_p_value: float | None, n_tests: int) -> float:
    if best_p_value is None or n_tests <= 0:
        return 0.0
    p_window = float(best_p_value)
    if p_window < 0 or p_window > 1:
        raise ValueError(f"FIMO p-value must be in [0, 1], received {p_window}.")
    p_sequence = 1.0 - (1.0 - p_window) ** int(n_tests)
    p_sequence = max(float(p_sequence), float(np.finfo(float).tiny))
    return float(-math.log10(p_sequence))


def _count_tests(*, sequence_length: int, motif_width: int, bidirectional: bool) -> int:
    windows = max(int(sequence_length) - int(motif_width) + 1, 0)
    strands = 2 if bidirectional else 1
    return windows * strands


def _safe_correlation(series_a: pd.Series, series_b: pd.Series, *, method: str) -> float | None:
    if len(series_a) < 2 or len(series_b) < 2:
        return None
    if series_a.nunique(dropna=True) < 2 or series_b.nunique(dropna=True) < 2:
        return None
    corr = series_a.astype(float).corr(series_b.astype(float), method=method)
    if corr is None or pd.isna(corr):
        return None
    return float(corr)


def build_fimo_concordance_table(
    *,
    points_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, PWM],
    bidirectional: bool,
    threshold: float,
    work_dir: Path,
    tool_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if points_df is None or points_df.empty:
        raise ValueError("Cannot compute FIMO concordance from an empty point table.")
    if "sequence" not in points_df.columns:
        raise ValueError("FIMO concordance requires trajectory points with a 'sequence' column.")
    if "objective_scalar" not in points_df.columns:
        raise ValueError("FIMO concordance requires trajectory points with an 'objective_scalar' column.")

    objective_vals = pd.to_numeric(points_df["objective_scalar"], errors="coerce")
    if objective_vals.isna().any():
        raise ValueError("FIMO concordance requires numeric objective_scalar values.")

    work = points_df.copy().reset_index(drop=True)
    if "phase" in work.columns:
        draw_only = work[work["phase"].astype(str) == "draw"].copy()
        if not draw_only.empty:
            work = draw_only.reset_index(drop=True)
    work["sequence"] = work["sequence"].astype(str)
    work["_seq_name"] = [f"seq_{idx:06d}" for idx in range(len(work))]
    work["objective_scalar"] = pd.to_numeric(work["objective_scalar"], errors="coerce").astype(float)

    work_dir.mkdir(parents=True, exist_ok=True)
    records = list(zip(work["_seq_name"].tolist(), work["sequence"].tolist()))
    fasta_path = work_dir / "fimo_candidates.fa"
    _write_candidates_fasta(records, fasta_path)

    tf_score_columns: list[str] = []
    for tf_name in tf_names:
        pwm = pwms.get(tf_name)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf_name}' while computing FIMO concordance.")
        motif_path = work_dir / f"{_sanitize_id(tf_name)}.meme"
        _write_minimal_meme_motif(pwm, motif_path)
        rows = _run_fimo(
            motif_path=motif_path,
            fasta_path=fasta_path,
            bidirectional=bidirectional,
            threshold=threshold,
            tool_path=tool_path,
        )
        best_p = _best_p_values_by_sequence(rows)
        scores: list[float] = []
        motif_width = int(pwm.length)
        for seq_name, seq in records:
            n_tests = _count_tests(sequence_length=len(seq), motif_width=motif_width, bidirectional=bidirectional)
            score = _sequence_level_fimo_score(best_p_value=best_p.get(seq_name), n_tests=n_tests)
            scores.append(score)
        score_col = f"fimo_score_{tf_name}"
        work[score_col] = scores
        tf_score_columns.append(score_col)

    if not tf_score_columns:
        raise ValueError("FIMO concordance requires at least one TF.")

    work["fimo_joint_weakest_score"] = work[tf_score_columns].min(axis=1).astype(float)

    concordance_df = work[
        ["objective_scalar", "fimo_joint_weakest_score", "sequence", "_seq_name", *tf_score_columns]
    ].copy()
    concordance_df = concordance_df.rename(columns={"_seq_name": "sequence_name"})

    pearson = _safe_correlation(
        concordance_df["objective_scalar"],
        concordance_df["fimo_joint_weakest_score"],
        method="pearson",
    )
    spearman = _safe_correlation(
        concordance_df["objective_scalar"],
        concordance_df["fimo_joint_weakest_score"],
        method="spearman",
    )
    summary = {
        "n_rows": int(len(concordance_df)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "threshold": float(threshold),
        "bidirectional": bool(bidirectional),
        "joint_metric": "min_tf(-log10 sequence-level FIMO p-value)",
    }
    return concordance_df, summary
