"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_artifact.py

PWM input source for per-motif JSON artifacts (contract-first).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites

_SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
_BASES = ("A", "C", "G", "T")


def _require(obj: dict, key: str, *, label: str) -> Any:
    if key not in obj:
        raise ValueError(f"PWM artifact missing required field: {label}.{key}")
    return obj[key]


def _as_float(value: Any, *, label: str) -> float:
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"PWM artifact {label} must be a number, got {value!r}") from e


def _validate_background(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("PWM artifact background must be an object with A/C/G/T keys.")
    bg = {}
    for base in _BASES:
        if base not in raw:
            raise ValueError(f"PWM artifact background missing base {base}.")
        val = _as_float(raw[base], label=f"background[{base}]")
        if val <= 0:
            raise ValueError(f"PWM artifact background[{base}] must be > 0.")
        bg[base] = val
    total = sum(bg.values())
    if total <= 0 or abs(total - 1.0) > 1e-3:
        raise ValueError("PWM artifact background must sum to 1.0 (within tolerance).")
    return normalize_background(bg)


def _validate_probability_rows(raw: Any) -> List[dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("PWM artifact probabilities must be a non-empty list.")
    rows: List[dict[str, float]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"PWM artifact probabilities[{idx}] must be an object.")
        vals: dict[str, float] = {}
        for base in _BASES:
            if base not in row:
                raise ValueError(f"PWM artifact probabilities[{idx}] missing base {base}.")
            val = _as_float(row[base], label=f"probabilities[{idx}][{base}]")
            if val < 0:
                raise ValueError(f"PWM artifact probabilities[{idx}][{base}] must be >= 0.")
            vals[base] = val
        total = sum(vals.values())
        if total <= 0:
            raise ValueError(f"PWM artifact probabilities[{idx}] sums to 0.")
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"PWM artifact probabilities[{idx}] must sum to 1.0 (within tolerance).")
        vals = {b: v / total for b, v in vals.items()}
        rows.append(vals)
    return rows


def _validate_log_odds_rows(raw: Any, *, expected_len: int) -> List[dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("PWM artifact log_odds must be a non-empty list.")
    if len(raw) != expected_len:
        raise ValueError(f"PWM artifact log_odds length ({len(raw)}) does not match probabilities ({expected_len}).")
    rows: List[dict[str, float]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"PWM artifact log_odds[{idx}] must be an object.")
        vals: dict[str, float] = {}
        for base in _BASES:
            if base not in row:
                raise ValueError(f"PWM artifact log_odds[{idx}] missing base {base}.")
            val = _as_float(row[base], label=f"log_odds[{idx}][{base}]")
            if not math.isfinite(val):
                raise ValueError(f"PWM artifact log_odds[{idx}][{base}] must be finite.")
            vals[base] = val
        rows.append(vals)
    return rows


def _load_artifact(path: Path) -> PWMMotif:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("PWM artifact must be a JSON object.")

    schema_version = _require(raw, "schema_version", label="root")
    if str(schema_version) not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported PWM artifact schema_version: {schema_version}")

    producer = str(_require(raw, "producer", label="root")).strip()
    if not producer:
        raise ValueError("PWM artifact producer must be a non-empty string.")

    motif_id = str(_require(raw, "motif_id", label="root")).strip()
    if not motif_id:
        raise ValueError("PWM artifact motif_id must be a non-empty string.")

    alphabet = str(_require(raw, "alphabet", label="root")).strip().upper()
    if alphabet != "ACGT":
        raise ValueError(f"PWM artifact alphabet must be ACGT, got {alphabet!r}")

    matrix_semantics = str(_require(raw, "matrix_semantics", label="root")).strip().lower()
    if matrix_semantics != "probabilities":
        raise ValueError(f"PWM artifact matrix_semantics must be 'probabilities', got {matrix_semantics!r}")

    background = _validate_background(_require(raw, "background", label="root"))
    probabilities = _validate_probability_rows(_require(raw, "probabilities", label="root"))
    log_odds = _validate_log_odds_rows(_require(raw, "log_odds", label="root"), expected_len=len(probabilities))

    declared_len = raw.get("length")
    if declared_len is not None:
        try:
            decl = int(declared_len)
        except Exception as e:
            raise ValueError("PWM artifact length must be an integer when provided.") from e
        if decl != len(probabilities):
            raise ValueError(f"PWM artifact length ({decl}) does not match probabilities ({len(probabilities)}).")

    return PWMMotif(motif_id=motif_id, matrix=probabilities, background=background, log_odds=log_odds)


def load_artifact(path: Path) -> PWMMotif:
    return _load_artifact(path)


@dataclass
class PWMArtifactDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    sampling: dict
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        artifact_path = resolve_path(self.cfg_path, self.path)
        if not (artifact_path.exists() and artifact_path.is_file()):
            raise FileNotFoundError(f"PWM artifact not found. Looked here:\n  - {artifact_path}")

        motif = _load_artifact(artifact_path)
        motif_hash = hash_pwm_motif(
            motif_label=motif.motif_id,
            matrix=motif.matrix,
            background=motif.background,
            source_kind="pwm_artifact",
        )

        sampling = dict(self.sampling or {})
        strategy = str(sampling.get("strategy", "stochastic"))
        n_sites = int(sampling.get("n_sites"))
        oversample_factor = int(sampling.get("oversample_factor", 10))
        max_candidates = sampling.get("max_candidates")
        max_seconds = sampling.get("max_seconds")
        threshold = sampling.get("score_threshold")
        percentile = sampling.get("score_percentile")
        length_policy = str(sampling.get("length_policy", "exact"))
        length_range = sampling.get("length_range")
        trim_window_length = sampling.get("trim_window_length")
        trim_window_strategy = sampling.get("trim_window_strategy", "max_info")
        scoring_backend = str(sampling.get("scoring_backend", "densegen")).lower()
        pvalue_threshold = sampling.get("pvalue_threshold")
        pvalue_bins = sampling.get("pvalue_bins")
        mining = sampling.get("mining")
        bgfile = sampling.get("bgfile")
        selection_policy = str(sampling.get("selection_policy", "random_uniform"))
        keep_all_candidates_debug = bool(sampling.get("keep_all_candidates_debug", False))
        include_matched_sequence = bool(sampling.get("include_matched_sequence", False))
        bgfile_path: Path | None = None
        if bgfile is not None:
            bgfile_path = resolve_path(self.cfg_path, str(bgfile))
            if not (bgfile_path.exists() and bgfile_path.is_file()):
                raise FileNotFoundError(f"Stage-A PWM sampling bgfile not found. Looked here:\n  - {bgfile_path}")
        debug_output_dir: Path | None = None
        if keep_all_candidates_debug:
            if outputs_root is None:
                raise ValueError("keep_all_candidates_debug requires outputs_root to be set.")
            if run_id is None:
                raise ValueError("keep_all_candidates_debug requires run_id to be set.")
            debug_output_dir = candidates_root(Path(outputs_root), run_id) / self.input_name

        return_meta = scoring_backend == "fimo"
        result = sample_pwm_sites(
            rng,
            motif,
            input_name=self.input_name,
            motif_hash=motif_hash,
            run_id=run_id,
            strategy=strategy,
            n_sites=n_sites,
            oversample_factor=oversample_factor,
            max_candidates=max_candidates,
            max_seconds=max_seconds,
            score_threshold=threshold,
            score_percentile=percentile,
            scoring_backend=scoring_backend,
            pvalue_threshold=pvalue_threshold,
            pvalue_bins=pvalue_bins,
            mining=mining,
            bgfile=bgfile_path,
            selection_policy=selection_policy,
            keep_all_candidates_debug=keep_all_candidates_debug,
            include_matched_sequence=include_matched_sequence,
            debug_output_dir=debug_output_dir,
            debug_label=f"{artifact_path.stem}__{motif.motif_id}",
            length_policy=length_policy,
            length_range=length_range,
            trim_window_length=trim_window_length,
            trim_window_strategy=str(trim_window_strategy),
            return_metadata=return_meta,
        )
        if return_meta:
            selected, meta_by_seq = result  # type: ignore[misc]
        else:
            selected = result  # type: ignore[assignment]
            meta_by_seq = {}

        entries = [(motif.motif_id, seq, str(artifact_path)) for seq in selected]
        import pandas as pd

        rows = []
        for seq in selected:
            meta = meta_by_seq.get(seq, {}) if meta_by_seq else {}
            start = meta.get("fimo_start")
            stop = meta.get("fimo_stop")
            strand = meta.get("fimo_strand")
            tfbs_id = hash_tfbs_id(
                motif_id=motif_hash,
                sequence=seq,
                scoring_backend=scoring_backend,
                matched_start=int(start) if start is not None else None,
                matched_stop=int(stop) if stop is not None else None,
                matched_strand=str(strand) if strand is not None else None,
            )
            row = {
                "tf": motif.motif_id,
                "tfbs": seq,
                "source": str(artifact_path),
                "motif_id": motif_hash,
                "tfbs_id": tfbs_id,
            }
            if meta:
                row.update(meta)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        return entries, df_out
