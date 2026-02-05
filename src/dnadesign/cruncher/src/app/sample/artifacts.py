"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/sample/artifacts.py

Persist sample workflow artifacts, configs, and run markers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dnadesign.cruncher.app.sample.resources import _lockmap_for, _store
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_yaml
from dnadesign.cruncher.artifacts.layout import config_used_path
from dnadesign.cruncher.config.schema_v2 import CruncherConfig, SampleConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.store.motif_store import MotifRef

logger = logging.getLogger(__name__)


def _save_config(
    cfg: CruncherConfig,
    batch_dir: Path,
    config_path: Path,
    *,
    tfs: list[str],
    set_index: int,
    sample_cfg: SampleConfig | None = None,
    log_fn: Callable[..., None] | None = None,
) -> None:
    """
    Save the exact Pydantic-validated config into <batch_dir>/meta/config_used.yaml,
    plus, for each TF:
      - alphabet: ["A","C","G","T"]
      - pwm_matrix: a list of [p_A, p_C, p_G, p_T] for each position
      - consensus: consensus sequence string
    """
    cfg_path = config_used_path(batch_dir)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json")
    if sample_cfg is not None:
        data["sample"] = sample_cfg.model_dump(mode="json")

    store = _store(cfg, config_path)
    pwms_info: dict[str, dict[str, object]] = {}

    logger.debug("Saving PWM info for config_used.yaml")
    lockmap = _lockmap_for(cfg, config_path)
    for tf in sorted(tfs):
        entry = lockmap.get(tf)
        if entry is None:
            raise ValueError(f"Missing lock entry for TF '{tf}'")
        ref = MotifRef(source=entry.source, motif_id=entry.motif_id)
        pwm: PWM = store.get_pwm(ref)

        # Build consensus: argmax over each column
        cons_vec = np.argmax(pwm.matrix, axis=1)
        consensus = "".join("ACGT"[i] for i in cons_vec)

        pwm_probs_rounded: list[list[float]] = []
        for row in pwm.matrix:
            rounded_row = [round(float(p), 6) for p in row]
            pwm_probs_rounded.append(rounded_row)

        pwms_info[tf] = {
            "alphabet": ["A", "C", "G", "T"],
            "pwm_matrix": pwm_probs_rounded,
            "consensus": consensus,
        }

    data["pwms_info"] = pwms_info
    data["active_regulator_set"] = {"index": set_index, "tfs": tfs}
    atomic_write_yaml(cfg_path, {"cruncher": data}, sort_keys=False, default_flow_style=False)
    log_fn = log_fn or logger.info
    log_fn("Wrote config_used.yaml to %s", cfg_path.relative_to(batch_dir.parent))


def _write_parquet_rows(
    path: Path,
    rows: Iterable[dict[str, object]],
    *,
    chunk_size: int = 10000,
    schema: Any | None = None,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, object]] = []
    count = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) < chunk_size:
                continue
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), table.schema)
            writer.write_table(table)
            count += len(buffer)
            buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), table.schema)
            writer.write_table(table)
            count += len(buffer)
        if writer is not None:
            writer.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(path)
        elif schema is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            empty = pa.Table.from_pylist([], schema=schema)
            pq.write_table(empty, str(tmp_path))
            tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return count


def _elite_parquet_schema(tfs: Iterable[str], *, include_canonical: bool) -> Any:
    import pyarrow as pa

    fields = [
        pa.field("sequence", pa.string()),
        pa.field("rank", pa.int64()),
        pa.field("norm_sum", pa.float64()),
        pa.field("min_norm", pa.float64()),
        pa.field("sum_norm", pa.float64()),
        pa.field("combined_score_final", pa.float64()),
        pa.field("chain", pa.int64()),
        pa.field("chain_1based", pa.int64()),
        pa.field("draw_idx", pa.int64()),
        pa.field("draw_in_phase", pa.int64()),
        pa.field("meta_type", pa.string()),
        pa.field("meta_source", pa.string()),
        pa.field("meta_date", pa.string()),
        pa.field("per_tf_json", pa.string()),
    ]
    if include_canonical:
        fields.append(pa.field("canonical_sequence", pa.string()))
    for tf_name in sorted(tfs):
        fields.append(pa.field(f"score_{tf_name}", pa.float64()))
        fields.append(pa.field(f"norm_{tf_name}", pa.float64()))
    return pa.schema(fields)


def _format_run_path(path: Path, *, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
