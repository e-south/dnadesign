"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_fimo.py

Helpers for MEME Suite FIMO-backed scoring of PWM-sampled candidates.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ...core.stage_a.stage_a_sampling_utils import normalize_background
from ...core.stage_a.stage_a_types import PWMMotif
from ...integrations.meme_suite import require_executable

_HEADER_RE = re.compile(r"[\s\-]+")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class FimoHit:
    sequence_name: str
    start: int
    stop: int
    strand: str
    score: float
    matched_sequence: str | None = None


def _normalize_header(name: str) -> str:
    return _HEADER_RE.sub("_", str(name).strip().lower())


def _sanitize_id(text: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", str(text).strip())
    return cleaned or "motif"


def build_candidate_records(
    motif_id: str,
    sequences: Sequence[str],
    *,
    start_index: int = 0,
) -> list[tuple[str, str]]:
    prefix = _sanitize_id(motif_id)
    return [(f"{prefix}|cand{start_index + idx}", seq) for idx, seq in enumerate(sequences)]


def write_candidates_fasta(records: Sequence[tuple[str, str]], out_path: Path) -> None:
    lines = []
    for rec_id, seq in records:
        lines.append(f">{rec_id}")
        lines.append(str(seq))
    out_path.write_text("\n".join(lines) + "\n")


def write_minimal_meme_motif(motif: PWMMotif, out_path: Path) -> str:
    motif_id = _sanitize_id(motif.motif_id)
    bg = normalize_background(motif.background)
    lines = [
        "MEME version 4",
        "",
        "ALPHABET= ACGT",
        "",
        "strands: + -",
        "",
        "Background letter frequencies:",
        f"A {bg['A']:.6g} C {bg['C']:.6g} G {bg['G']:.6g} T {bg['T']:.6g}",
        "",
        f"MOTIF {motif_id}",
        f"letter-probability matrix: alength= 4 w= {len(motif.matrix)}",
    ]
    for row in motif.matrix:
        lines.append(
            f"{float(row.get('A', 0.0)):.6g} {float(row.get('C', 0.0)):.6g} "
            f"{float(row.get('G', 0.0)):.6g} {float(row.get('T', 0.0)):.6g}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    return motif_id


def parse_fimo_tsv(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader, None)
    if header is None:
        return []
    alias = {"pvalue": "p_value", "qvalue": "q_value", "sequence": "sequence_name"}
    normalized = [alias.get(_normalize_header(h), _normalize_header(h)) for h in header]
    idx = {name: i for i, name in enumerate(normalized)}
    required = {"sequence_name", "start", "stop", "strand", "score"}
    if not required.issubset(idx):
        raise ValueError(f"FIMO output missing required columns: {sorted(required - set(idx))}")
    rows: list[dict] = []
    for row in reader:
        if not row:
            continue
        seq_name = row[idx["sequence_name"]]
        entry = {
            "sequence_name": seq_name,
            "start": int(row[idx["start"]]),
            "stop": int(row[idx["stop"]]),
            "strand": row[idx["strand"]],
            "score": float(row[idx["score"]]),
        }
        if "p_value" in idx:
            entry["p_value"] = float(row[idx["p_value"]])
        if "q_value" in idx:
            try:
                entry["q_value"] = float(row[idx["q_value"]])
            except Exception:
                entry["q_value"] = None
        if "matched_sequence" in idx:
            entry["matched_sequence"] = row[idx["matched_sequence"]]
        rows.append(entry)
    return rows


def aggregate_best_hits(rows: Iterable[dict], *, allowed_strands: set[str] | None = None) -> dict[str, FimoHit]:
    best: dict[str, FimoHit] = {}
    strand_filter = allowed_strands if allowed_strands is not None else {"+"}
    for row in rows:
        strand = str(row.get("strand"))
        if strand_filter and strand not in strand_filter:
            continue
        seq_name = row["sequence_name"]
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"FIMO hit has non-finite score for '{seq_name}'.")
        hit = FimoHit(
            sequence_name=seq_name,
            start=int(row["start"]),
            stop=int(row["stop"]),
            strand=strand,
            score=score,
            matched_sequence=row.get("matched_sequence"),
        )
        prev = best.get(seq_name)
        if prev is None:
            best[seq_name] = hit
            continue
        if score > prev.score:
            best[seq_name] = hit
            continue
        if score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop):
            best[seq_name] = hit
    return best


def run_fimo(
    *,
    meme_motif_path: Path,
    fasta_path: Path,
    bgfile: Path | str | None = None,
    norc: bool = False,
    thresh: float | None = None,
    include_matched_sequence: bool = False,
    return_tsv: bool = False,
) -> tuple[list[dict], str | None]:
    exe = require_executable("fimo", tool_path=None)
    cmd = [str(exe), "--text"]
    if not include_matched_sequence:
        cmd.append("--skip-matched-sequence")
    if norc:
        cmd.append("--norc")
    if thresh is not None:
        cmd.extend(["--thresh", str(thresh)])
    if bgfile is not None:
        cmd.extend(["--bgfile", str(bgfile)])
    cmd.extend([str(meme_motif_path), str(fasta_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"FIMO failed (exit {result.returncode}). {stderr or 'No stderr output.'}")
    tsv_text = result.stdout
    rows = parse_fimo_tsv(tsv_text)
    return rows, (tsv_text if return_tsv else None)
