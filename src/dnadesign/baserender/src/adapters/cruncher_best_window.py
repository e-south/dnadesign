"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/cruncher_best_window.py

Cruncher adapter mapping elite best-window hits into kmer features and motif_logo effects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..core import Record, SchemaError, SkipRecord, Span
from ..core.record import Display, Effect, Feature, revcomp

_CRUNCHER_POLICY_DEFAULTS: dict[str, str] = {
    "on_missing_hit": "error",
    "on_missing_pwm": "error",
}


@dataclass(frozen=True)
class _SamplingWindow:
    sequence_length: int
    maxw: int | None
    strategy: str

    @property
    def max_allowed(self) -> int:
        return self.sequence_length if self.maxw is None else min(self.maxw, self.sequence_length)


def _resolve_sampling_window(cruncher_cfg: Mapping[str, Any]) -> _SamplingWindow | None:
    sample_cfg = cruncher_cfg.get("sample")
    if not isinstance(sample_cfg, Mapping):
        return None

    motif_width = sample_cfg.get("motif_width")
    if motif_width is None:
        return None
    if not isinstance(motif_width, Mapping):
        raise SchemaError("cruncher_best_window config_path sample.motif_width must be a mapping")

    seq_len = sample_cfg.get("sequence_length")
    if not isinstance(seq_len, int) or seq_len < 1:
        raise SchemaError("cruncher_best_window config_path sample.sequence_length must be an integer >= 1")

    maxw_raw = motif_width.get("maxw")
    if maxw_raw is None:
        maxw = None
    else:
        try:
            maxw = int(maxw_raw)
        except Exception as exc:
            raise SchemaError(
                "cruncher_best_window config_path sample.motif_width.maxw must be an integer >= 1"
            ) from exc
        if maxw < 1:
            raise SchemaError("cruncher_best_window config_path sample.motif_width.maxw must be >= 1")

    strategy = str(motif_width.get("strategy") or "max_info").strip()
    if strategy != "max_info":
        raise SchemaError("cruncher_best_window config_path sample.motif_width.strategy must be 'max_info'")
    return _SamplingWindow(sequence_length=seq_len, maxw=maxw, strategy=strategy)


def _select_max_info_window(matrix: list[list[float]], *, length: int) -> list[list[float]]:
    if length < 1:
        raise SchemaError("cruncher_best_window PWM window length must be >= 1")
    if length > len(matrix):
        raise SchemaError(f"cruncher_best_window PWM window length ({length}) exceeds matrix length ({len(matrix)})")
    if length == len(matrix):
        return matrix

    arr = np.asarray(matrix, dtype=float)
    best_start = 0
    best_score = float("-inf")
    last_start = arr.shape[0] - length
    for start in range(last_start + 1):
        window = arr[start : start + length]
        p = window + 1e-9
        score = float((2 + (p * np.log2(p)).sum(axis=1)).sum())
        if score > best_score:
            best_score = score
            best_start = start
    return [[float(x) for x in row] for row in arr[best_start : best_start + length].tolist()]


@dataclass(frozen=True)
class CruncherBestWindowAdapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str
    hits_by_elite_id: Mapping[str, list[dict[str, Any]]]
    pwm_by_tf: Mapping[str, list[list[float]]]

    def __post_init__(self) -> None:
        merged = dict(_CRUNCHER_POLICY_DEFAULTS)
        merged.update({str(k): str(v) for k, v in dict(self.policies or {}).items()})
        object.__setattr__(self, "policies", merged)

    @classmethod
    def from_config(
        cls, *, columns: Mapping[str, Any], policies: Mapping[str, Any], alphabet: str
    ) -> "CruncherBestWindowAdapter":
        hits_path = Path(str(columns["hits_path"]))
        config_path = Path(str(columns["config_path"]))

        if not hits_path.exists():
            raise SchemaError(f"cruncher_best_window hits_path does not exist: {hits_path}")
        if not config_path.exists():
            raise SchemaError(f"cruncher_best_window config_path does not exist: {config_path}")

        import pyarrow.parquet as pq

        hits_table = pq.read_table(hits_path)
        hits_rows = hits_table.to_pylist()

        elite_col = str(columns.get("hits_elite_id", "elite_id"))
        tf_col = str(columns.get("hits_tf", "tf"))
        start_col = str(columns.get("hits_start", "best_start"))
        strand_col = str(columns.get("hits_strand", "best_strand"))
        window_col = str(columns.get("hits_window_seq", "best_window_seq"))
        core_col = str(columns.get("hits_core_seq", "best_core_seq"))

        required = {elite_col, tf_col, start_col, strand_col, window_col}
        row_keys = set(hits_table.schema.names)
        missing = sorted(required - row_keys)
        if missing:
            raise SchemaError(f"cruncher_best_window hits parquet missing required columns: {missing}")

        hits_by_elite_id: dict[str, list[dict[str, Any]]] = {}
        for row in hits_rows:
            rid = row.get(elite_col)
            if rid is None:
                continue
            hit = {
                "tf": row.get(tf_col),
                "start": row.get(start_col),
                "strand": row.get(strand_col),
                "window_seq": row.get(window_col),
                "core_seq": row.get(core_col),
            }
            hits_by_elite_id.setdefault(str(rid), []).append(hit)

        payload = yaml.safe_load(config_path.read_text()) or {}
        cruncher = payload.get("cruncher")
        if not isinstance(cruncher, Mapping):
            raise SchemaError("cruncher_best_window config_path missing top-level 'cruncher' mapping")
        pwms_info = cruncher.get("pwms_info")
        if not isinstance(pwms_info, Mapping):
            raise SchemaError("cruncher_best_window config_path missing 'cruncher.pwms_info'")
        sampling_window = _resolve_sampling_window(cruncher)

        pwm_by_tf: dict[str, list[list[float]]] = {}
        for tf, info in pwms_info.items():
            if not isinstance(info, Mapping):
                continue
            matrix = info.get("pwm_matrix")
            if isinstance(matrix, list) and matrix:
                parsed_matrix = [[float(x) for x in row] for row in matrix]
                if sampling_window is not None and len(parsed_matrix) > sampling_window.max_allowed:
                    parsed_matrix = _select_max_info_window(parsed_matrix, length=sampling_window.max_allowed)
                pwm_by_tf[str(tf)] = parsed_matrix

        if not pwm_by_tf:
            raise SchemaError("cruncher_best_window could not load any PWM matrices from config_path")

        return cls(
            columns=columns,
            policies=policies,
            alphabet=alphabet,
            hits_by_elite_id=hits_by_elite_id,
            pwm_by_tf=pwm_by_tf,
        )

    def apply(self, row: dict, *, row_index: int) -> Record:
        seq_col = str(self.columns.get("sequence", "sequence"))
        id_col = str(self.columns.get("id", "id"))

        sequence_raw = row.get(seq_col)
        if sequence_raw is None or str(sequence_raw).strip() == "":
            raise SchemaError(f"cruncher_best_window row missing sequence column '{seq_col}'")
        sequence = str(sequence_raw)

        rid_raw = row.get(id_col)
        if rid_raw is None or str(rid_raw).strip() == "":
            raise SchemaError(f"cruncher_best_window row missing id column '{id_col}'")
        record_id = str(rid_raw)

        hits = self.hits_by_elite_id.get(record_id, [])
        if not hits:
            policy = str(self.policies["on_missing_hit"]).lower()
            if policy == "skip":
                raise SkipRecord(f"cruncher_best_window no hits for elite_id={record_id}")
            raise SchemaError(f"cruncher_best_window no hits found for elite_id={record_id}")

        features: list[Feature] = []
        effects: list[Effect] = []
        tag_labels: dict[str, str] = {}

        for idx, hit in enumerate(hits):
            tf = str(hit.get("tf") or "").strip()
            if tf == "":
                raise SchemaError(f"cruncher_best_window hit missing tf for elite_id={record_id}")

            start_raw = hit.get("start")
            if start_raw is None:
                raise SchemaError(f"cruncher_best_window hit missing start for elite_id={record_id}, tf={tf}")
            start = int(start_raw)

            strand_raw = str(hit.get("strand") or "+").strip()
            if strand_raw == "+":
                strand = "fwd"
            elif strand_raw == "-":
                strand = "rev"
            else:
                raise SchemaError(
                    f"cruncher_best_window hit has invalid strand for elite_id={record_id}, tf={tf}: {strand_raw!r}"
                )

            window = str(hit.get("window_seq") or "").strip().upper()
            if window == "":
                raise SchemaError(f"cruncher_best_window hit missing window_seq for elite_id={record_id}, tf={tf}")
            core_seq = str(hit.get("core_seq") or "").strip().upper()
            if strand == "rev":
                label = core_seq if core_seq else revcomp(window)
            else:
                label = window

            span = Span(start=start, end=start + len(window), strand=strand)
            feature_id = f"{record_id}:best_window:{tf}:{idx}"
            tag = f"tf:{tf}"
            features.append(
                Feature(
                    id=feature_id,
                    kind="kmer",
                    span=span,
                    label=label,
                    tags=(tag,),
                    attrs={"tf": tf, "source": "cruncher_best_window"},
                    render={"priority": 10},
                )
            )
            tag_labels.setdefault(tag, tf)

            matrix = self.pwm_by_tf.get(tf)
            if matrix is None:
                policy = str(self.policies["on_missing_pwm"]).lower()
                if policy == "skip_effect":
                    continue
                raise SchemaError(f"cruncher_best_window missing PWM matrix for tf='{tf}'")
            if len(matrix) != len(window):
                raise SchemaError(
                    "cruncher_best_window PWM matrix length mismatch for tf="
                    f"'{tf}': matrix={len(matrix)} hit_window={len(window)} elite_id={record_id}"
                )

            effects.append(
                Effect(
                    kind="motif_logo",
                    target={"feature_id": feature_id},
                    params={"matrix": matrix},
                    render={"priority": 20},
                )
            )

        record = Record(
            id=record_id,
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(overlay_text=None, tag_labels=tag_labels),
            meta={"row_index": row_index, "adapter": "cruncher_best_window"},
        )
        return record.validate()
