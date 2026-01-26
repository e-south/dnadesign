"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_jaspar.py

PWM input source (JASPAR PFM format) with explicit Stage-A sampling policies.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_jaspar(path: Path) -> List[PWMMotif]:
    motifs: List[PWMMotif] = []
    current_id: Optional[str] = None
    rows: dict[str, List[float]] = {}

    def _finalize():
        nonlocal current_id, rows
        if current_id is None:
            return
        for base in ("A", "C", "G", "T"):
            if base not in rows:
                raise ValueError(f"JASPAR motif {current_id} missing row for {base}")
        lengths = {len(rows[b]) for b in ("A", "C", "G", "T")}
        if len(lengths) != 1:
            raise ValueError(f"JASPAR motif {current_id} has inconsistent row lengths: {sorted(lengths)}")
        width = lengths.pop()
        if width <= 0:
            raise ValueError(f"JASPAR motif {current_id} has zero width.")
        matrix = []
        for i in range(width):
            a, c, g, t = rows["A"][i], rows["C"][i], rows["G"][i], rows["T"][i]
            total = a + c + g + t
            if total <= 0:
                raise ValueError(f"JASPAR motif {current_id} has zero-sum column at {i}.")
            matrix.append({"A": a / total, "C": c / total, "G": g / total, "T": t / total})
        motifs.append(PWMMotif(motif_id=current_id, matrix=matrix, background=normalize_background(None)))
        current_id = None
        rows = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _finalize()
            parts = line[1:].strip().split()
            if not parts:
                raise ValueError("JASPAR motif header missing ID.")
            current_id = parts[0].strip()
            rows = {}
            continue
        if current_id is None:
            continue
        base = line[0].upper()
        if base not in {"A", "C", "G", "T"}:
            raise ValueError(f"Unexpected row label '{base}' in JASPAR motif {current_id}")
        nums = _NUM_RE.findall(line[1:])
        if not nums:
            raise ValueError(f"JASPAR row for {current_id} ({base}) is empty.")
        rows[base] = [float(x) for x in nums]

    _finalize()
    if not motifs:
        raise ValueError("No motifs found in JASPAR file.")
    return motifs


@dataclass
class PWMJasparDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_ids: Optional[List[str]]
    sampling: dict
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        jaspar_path = resolve_path(self.cfg_path, self.path)
        if not (jaspar_path.exists() and jaspar_path.is_file()):
            raise FileNotFoundError(f"PWM JASPAR file not found. Looked here:\n  - {jaspar_path}")

        motifs = _parse_jaspar(jaspar_path)
        if self.motif_ids:
            keep = set(self.motif_ids)
            motifs = [m for m in motifs if m.motif_id in keep]
            if not motifs:
                raise ValueError(f"No motifs matched motif_ids in {jaspar_path}")

        sampling = dict(self.sampling or {})
        strategy = str(sampling.get("strategy", "stochastic"))
        n_sites = int(sampling.get("n_sites"))
        oversample_factor = int(sampling.get("oversample_factor", 10))
        length_policy = str(sampling.get("length_policy", "exact"))
        length_range = sampling.get("length_range")
        trim_window_length = sampling.get("trim_window_length")
        trim_window_strategy = sampling.get("trim_window_strategy", "max_info")
        scoring_backend = str(sampling.get("scoring_backend", "fimo")).lower()
        mining = sampling.get("mining")
        bgfile = sampling.get("bgfile")
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

        entries = []
        all_rows = []
        summaries = []
        for motif in motifs:
            motif_hash = hash_pwm_motif(
                motif_label=motif.motif_id,
                matrix=motif.matrix,
                background=motif.background,
                source_kind="pwm_jaspar",
            )
            return_meta = True
            result = sample_pwm_sites(
                rng,
                motif,
                input_name=self.input_name,
                motif_hash=motif_hash,
                run_id=run_id,
                strategy=strategy,
                n_sites=n_sites,
                oversample_factor=oversample_factor,
                scoring_backend=scoring_backend,
                mining=mining,
                bgfile=bgfile_path,
                keep_all_candidates_debug=keep_all_candidates_debug,
                include_matched_sequence=include_matched_sequence,
                debug_output_dir=debug_output_dir,
                debug_label=f"{jaspar_path.stem}__{motif.motif_id}",
                length_policy=length_policy,
                length_range=length_range,
                trim_window_length=trim_window_length,
                trim_window_strategy=str(trim_window_strategy),
                return_metadata=return_meta,
                return_summary=True,
            )
            if return_meta:
                selected, meta_by_seq, summary = result  # type: ignore[misc]
            else:
                selected, summary = result  # type: ignore[assignment]
                meta_by_seq = {}
            if summary is not None:
                summaries.append(summary)
            for seq in selected:
                entries.append((motif.motif_id, seq, str(jaspar_path)))
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
                    "regulator_id": motif.motif_id,
                    "tfbs_sequence": seq,
                    "source": str(jaspar_path),
                    "motif_id": motif_hash,
                    "tfbs_id": tfbs_id,
                }
                if meta:
                    row.update(meta)
                all_rows.append(row)

        import pandas as pd

        df = pd.DataFrame(all_rows)
        return entries, df, summaries
