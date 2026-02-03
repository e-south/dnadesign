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

from ...config import PWMSamplingConfig
from ...core.artifacts.ids import hash_pwm_motif, hash_tfbs_id
from ...core.run_paths import candidates_root
from .base import BaseDataSource, resolve_path
from .pwm_sampling import sample_pwm_sites, sampling_kwargs_from_config, validate_mmr_core_length
from .stage_a_sampling_utils import normalize_background
from .stage_a_types import PWMMotif


@dataclass
class PWMMatrixCSVDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    motif_id: str
    columns: dict[str, str]
    sampling: PWMSamplingConfig
    input_name: str

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        if rng is None:
            raise ValueError("Stage-A PWM sampling requires an RNG; pass the pipeline RNG explicitly.")
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
        motif_hash = hash_pwm_motif(
            motif_label=motif.motif_id,
            matrix=motif.matrix,
            background=motif.background,
            source_kind="pwm_matrix_csv",
        )

        sampling_kwargs = sampling_kwargs_from_config(self.sampling)
        selection_cfg = sampling_kwargs.get("selection")
        selection_policy = str(getattr(selection_cfg, "policy", None) or "top_score")
        validate_mmr_core_length(
            motif_id=motif.motif_id,
            motif_width=len(motif.matrix),
            selection_policy=selection_policy,
            length_policy=str(sampling_kwargs.get("length_policy") or "exact"),
            length_range=sampling_kwargs.get("length_range"),
            trim_window_length=sampling_kwargs.get("trim_window_length"),
        )
        bgfile = sampling_kwargs.get("bgfile")
        keep_all_candidates_debug = bool(sampling_kwargs.get("keep_all_candidates_debug", False))
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

        return_meta = True
        result = sample_pwm_sites(
            rng,
            motif,
            input_name=self.input_name,
            motif_hash=motif_hash,
            run_id=run_id,
            mining=sampling_kwargs["mining"],
            bgfile=bgfile_path,
            keep_all_candidates_debug=keep_all_candidates_debug,
            include_matched_sequence=sampling_kwargs["include_matched_sequence"],
            uniqueness_key=sampling_kwargs["uniqueness_key"],
            selection=sampling_kwargs["selection"],
            debug_output_dir=debug_output_dir,
            debug_label=f"{csv_path.stem}__{motif.motif_id}",
            length_policy=sampling_kwargs["length_policy"],
            length_range=sampling_kwargs["length_range"],
            trim_window_length=sampling_kwargs["trim_window_length"],
            trim_window_strategy=str(sampling_kwargs["trim_window_strategy"]),
            return_metadata=return_meta,
            return_summary=True,
            strategy=str(sampling_kwargs["strategy"]),
            n_sites=int(sampling_kwargs["n_sites"]),
        )
        if return_meta:
            selected, meta_by_seq, summary = result  # type: ignore[misc]
        else:
            selected, summary = result  # type: ignore[assignment]
            meta_by_seq = {}

        entries = [(motif.motif_id, seq, str(csv_path)) for seq in selected]
        rows = []
        for seq in selected:
            meta = meta_by_seq[seq] if return_meta else None
            start = meta.fimo_start if meta is not None else None
            stop = meta.fimo_stop if meta is not None else None
            strand = meta.fimo_strand if meta is not None else None
            tfbs_id = hash_tfbs_id(
                motif_id=motif_hash,
                sequence=seq,
                scoring_backend="fimo",
                matched_start=int(start) if start is not None else None,
                matched_stop=int(stop) if stop is not None else None,
                matched_strand=str(strand) if strand is not None else None,
            )
            row = {
                "tf": motif.motif_id,
                "tfbs": seq,
                "regulator_id": motif.motif_id,
                "tfbs_sequence": seq,
                "source": str(csv_path),
                "motif_id": motif_hash,
                "tfbs_id": tfbs_id,
            }
            if meta is not None:
                row.update(meta.to_dict())
            rows.append(row)
        df_out = pd.DataFrame(rows)
        summaries = [summary] if summary is not None else []
        return entries, df_out, summaries
