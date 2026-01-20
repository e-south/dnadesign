"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_matrix_csv.py

PWM input source (CSV matrix with A/C/G/T columns).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .base import BaseDataSource, resolve_path
from .pwm_sampling import PWMMotif, normalize_background, sample_pwm_sites


@dataclass
class PWMMatrixCSVDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_id: str
    columns: dict[str, str]
    sampling: dict

    def load_data(self, *, rng=None, outputs_root: Path | None = None):
        if rng is None:
            raise ValueError("PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
        if not self.motif_id or not str(self.motif_id).strip():
            raise ValueError("pwm_matrix_csv.motif_id must be a non-empty string")

        csv_path = resolve_path(self.cfg_path, self.path)
        if not (csv_path.exists() and csv_path.is_file()):
            raise FileNotFoundError(f"PWM matrix CSV not found. Looked here:\n  - {csv_path}")

        df = pd.read_csv(csv_path)
        cols = {k.upper(): v for k, v in self.columns.items()}
        required = {b: cols.get(b, b) for b in ("A", "C", "G", "T")}
        missing = [name for name in required.values() if name not in df.columns]
        if missing:
            raise ValueError(f"PWM matrix CSV missing columns: {missing}")

        matrix = []
        for i, row in df.iterrows():
            vals = {}
            for base, col in required.items():
                val = row[col]
                try:
                    num = float(val)
                except Exception as e:
                    raise ValueError(f"PWM matrix CSV row {i} has non-numeric {base} value: {val}") from e
                if num < 0:
                    raise ValueError(f"PWM matrix CSV row {i} has negative {base} value: {num}")
                vals[base] = num
            total = sum(vals.values())
            if total <= 0:
                raise ValueError(f"PWM matrix CSV row {i} sums to 0.")
            matrix.append({b: v / total for b, v in vals.items()})

        motif = PWMMotif(motif_id=str(self.motif_id).strip(), matrix=matrix, background=normalize_background(None))

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
                raise FileNotFoundError(f"PWM sampling bgfile not found. Looked here:\n  - {bgfile_path}")
        debug_output_dir: Path | None = None
        if keep_all_candidates_debug and outputs_root is not None:
            debug_output_dir = Path(outputs_root) / "meta" / "fimo"

        return_meta = scoring_backend == "fimo"
        result = sample_pwm_sites(
            rng,
            motif,
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
            debug_label=f"{csv_path.stem}__{motif.motif_id}",
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

        entries = [(motif.motif_id, seq, str(csv_path)) for seq in selected]
        rows = []
        for seq in selected:
            row = {"tf": motif.motif_id, "tfbs": seq, "source": str(csv_path)}
            if meta_by_seq:
                row.update(meta_by_seq.get(seq, {}))
            rows.append(row)
        df_out = pd.DataFrame(rows)
        return entries, df_out
