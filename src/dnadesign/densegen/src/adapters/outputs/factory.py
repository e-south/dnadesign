"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/factory.py

Factory for DenseGen output sinks.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ...config import DenseGenConfig, resolve_outputs_scoped_path, resolve_run_root
from ...core.run_paths import id_index_path
from .base import DEFAULT_NAMESPACE, SinkBase, USRSink
from .parquet import ParquetSink


def resolve_bio_alphabet(cfg: DenseGenConfig) -> tuple[str, str]:
    schema = cfg.output.output_schema
    return schema.bio_type, schema.alphabet


def build_sinks(cfg: DenseGenConfig, cfg_path: Path) -> Iterable[SinkBase]:
    """
    Create one or more sinks based on config:
      output.targets: usr/parquet
      output.usr / output.parquet
    """
    out_cfg = cfg.output
    run_root = resolve_run_root(cfg_path, cfg.run.root)
    targets = list(out_cfg.targets)
    sinks: list[SinkBase] = []

    default_bio, default_alpha = resolve_bio_alphabet(cfg)

    if "usr" in targets:
        usr_cfg = out_cfg.usr
        if usr_cfg is None:
            raise ValueError("output.usr is required when output.targets includes 'usr'")
        root = resolve_outputs_scoped_path(cfg_path, run_root, usr_cfg.root, label="output.usr.root")
        from .usr_writer import USRWriter

        writer = USRWriter(
            dataset=usr_cfg.dataset,
            root=root,
            namespace=DEFAULT_NAMESPACE,
            chunk_size=int(usr_cfg.chunk_size),
            health_event_interval_seconds=float(usr_cfg.health_event_interval_seconds),
            allow_overwrite=bool(usr_cfg.allow_overwrite),
            default_bio_type=default_bio,
            default_alphabet=default_alpha,
            npz_fields=list(usr_cfg.npz_fields),
            npz_root=usr_cfg.npz_root,
            run_quota=int(cfg.generation.total_quota()),
        )
        sinks.append(USRSink(writer))

    if "parquet" in targets:
        pq_cfg = out_cfg.parquet
        if pq_cfg is None:
            raise ValueError("output.parquet is required when output.targets includes 'parquet'")
        path = resolve_outputs_scoped_path(cfg_path, run_root, pq_cfg.path, label="output.parquet.path")
        ns = DEFAULT_NAMESPACE
        sinks.append(
            ParquetSink(
                path=str(path),
                namespace=ns,
                bio_type=default_bio,
                alphabet=default_alpha,
                deduplicate=pq_cfg.deduplicate,
                chunk_size=int(pq_cfg.chunk_size),
                index_path=id_index_path(run_root),
            )
        )

    if not sinks:
        raise AssertionError("No output sinks created; check output.targets and related settings.")

    return sinks
