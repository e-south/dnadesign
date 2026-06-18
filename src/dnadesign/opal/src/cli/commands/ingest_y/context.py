"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/ingest_y/context.py

Campaign and records context for `opal ingest-y`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ....core.utils import OpalError
from ....runtime.ingest_runtime import IngestRuntimeContract, build_ingest_runtime_contract
from ....storage.data_access import RecordsStore
from ....storage.label_sources import SharedObservedLabelSource, label_source_from_config
from .._common import load_cli_config, resolve_config_path, store_from_cfg

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class IngestCommandContext:
    cfg_path: Path
    cfg: Any
    store: RecordsStore
    label_source: Any
    records_df: "pd.DataFrame"
    ingest_runtime: IngestRuntimeContract
    shared_label_source: bool


def normalize_unknown_sequences_policy(value: str | None) -> str:
    policy = (value or "create").strip().lower()
    if policy not in {"create", "drop", "error"}:
        raise OpalError("--unknown-sequences must be one of: create, drop, error.")
    return policy


def build_ingest_command_context(config: Path | None, *, unknown_sequences: str) -> IngestCommandContext:
    cfg_path = resolve_config_path(config)
    cfg = load_cli_config(cfg_path)
    store: RecordsStore = store_from_cfg(cfg)
    label_source = label_source_from_config(cfg, store)
    shared_label_source = isinstance(label_source, SharedObservedLabelSource)
    records_df = store.load_ingest_identity_frame() if shared_label_source else store.load()
    ingest_runtime = build_ingest_runtime_contract(
        frame=records_df,
        records_path=store.records_path,
        records_row_count=store.row_count(),
        candidate_x_column=cfg.data.x_column_name,
        label_source_kind=getattr(label_source, "kind", "campaign_history"),
        fixed_candidate_universe=shared_label_source,
        unknown_sequences_policy=unknown_sequences,
    )
    return IngestCommandContext(
        cfg_path=cfg_path,
        cfg=cfg,
        store=store,
        label_source=label_source,
        records_df=records_df,
        ingest_runtime=ingest_runtime,
        shared_label_source=shared_label_source,
    )
