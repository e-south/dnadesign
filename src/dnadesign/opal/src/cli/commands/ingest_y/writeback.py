"""
Writeback operations for `opal ingest-y`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ....core.utils import OpalError, print_stdout
from ....storage.ledger import LedgerWriter
from ....storage.locks import CampaignLock
from ....storage.workspace import CampaignWorkspace
from ....storage.writebacks import build_label_events


@dataclass(frozen=True)
class IngestCommitResult:
    round_index: int
    labels_appended: int
    labels_skipped: int
    y_column_updated: str

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "ok": True,
            "applied": True,
            "round": self.round_index,
            "labels_appended": self.labels_appended,
            "labels_skipped": self.labels_skipped,
            "y_column_updated": self.y_column_updated,
        }


def commit_ingest_labels(
    *,
    cfg: Any,
    cfg_path: Path,
    store: Any,
    label_source: Any,
    records_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    required_cols: list[str],
    round_index: int,
    if_exists: str,
    shared_label_source: bool,
) -> IngestCommitResult:
    with CampaignLock(Path(cfg.campaign.workdir)):
        if not shared_label_source:
            records_df = store.ensure_rows_exist(
                records_df,
                labels_df,
                csv_df,
                required_cols=required_cols,
                conflict_policy=cfg.safety.conflict_policy_on_duplicate_ids,
            )

        labels_df = _resolve_missing_label_ids(records_df=records_df, labels_df=labels_df)
        existing_ids = _existing_ids_at_round(
            store=store,
            records_df=records_df,
            labels_df=labels_df,
            round_index=round_index,
            if_exists=if_exists,
            shared_label_source=shared_label_source,
        )
        labels_effective = _labels_after_if_exists_policy(
            labels_df=labels_df,
            if_exists=if_exists,
            existing_ids=existing_ids,
        )
        y_column_updated = cfg.data.y_column_name
        if shared_label_source:
            labels_effective = label_source.store.append_labels(
                labels_effective[["id", "y"]],
                observed_round=int(round_index),
                batch_id=f"round_{int(round_index)}",
                src="ingest_y",
                if_exists=str(if_exists).lower().strip(),
                known_ids=set(records_df["id"].astype(str).tolist()),
            )
            records_after = records_df
            y_column_updated = f"label_source:{label_source.store.path}"
        else:
            records_after = store.append_labels_from_df(
                records_df,
                labels_effective[["id", "y"]],
                r=int(round_index),
                src="ingest_y",
                fail_if_any_existing_labels=(str(if_exists).lower().strip() == "fail"),
                if_exists=str(if_exists).lower().strip(),
            )
            records_current = store.upsert_current_y_column(
                records_after,
                labels_effective[["id", "y"]],
                cfg.data.y_column_name,
            )
            store.save_atomic(records_current)

        _append_label_events(
            cfg=cfg,
            cfg_path=cfg_path,
            records_df=records_after,
            labels_effective=labels_effective,
            round_index=round_index,
        )

    return IngestCommitResult(
        round_index=int(round_index),
        labels_appended=int(len(labels_effective)),
        labels_skipped=int(len(labels_df) - len(labels_effective)),
        y_column_updated=y_column_updated,
    )


def _resolve_missing_label_ids(*, records_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if not labels_df["id"].isna().any():
        return labels_df
    seq_to_id = (
        records_df.set_index("sequence")["id"].astype(str).to_dict()
        if "sequence" in records_df.columns and "id" in records_df.columns
        else {}
    )
    labels_df = labels_df.copy()
    miss = labels_df["id"].isna()
    labels_df.loc[miss, "id"] = labels_df.loc[miss, "sequence"].map(seq_to_id)
    if labels_df["id"].isna().any():
        raise OpalError("Failed to resolve ids for some labels; provide id or ensure sequences exist.")
    return labels_df


def _existing_ids_at_round(
    *,
    store: Any,
    records_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    round_index: int,
    if_exists: str,
    shared_label_source: bool,
) -> set[str]:
    existing_ids: set[str] = set()
    if shared_label_source:
        return existing_ids
    try:
        label_hist_column = store.label_hist_col()
        ids_in = set(labels_df["id"].dropna().astype(str))
        if not ids_in:
            return existing_ids
        maybe = records_df.loc[records_df["id"].astype(str).isin(ids_in), ["id", label_hist_column]]
        duplicate_count = 0
        for label_id, cell in maybe.itertuples(index=False, name=None):
            for event in store._normalize_hist_cell(cell):
                if event.get("kind") == "label" and int(event.get("observed_round", -1)) == int(round_index):
                    duplicate_count += 1
                    existing_ids.add(str(label_id))
                    break
        if duplicate_count > 0:
            print_stdout(
                f"[notice] {duplicate_count}/{len(ids_in)} incoming labels already have r={int(round_index)};"
                f"applying --if-exists={if_exists}."
            )
    except Exception:
        pass
    return existing_ids


def _labels_after_if_exists_policy(
    *,
    labels_df: pd.DataFrame,
    if_exists: str,
    existing_ids: set[str],
) -> pd.DataFrame:
    if str(if_exists).lower().strip() == "skip" and existing_ids:
        return labels_df.loc[~labels_df["id"].astype(str).isin(existing_ids)].copy()
    return labels_df


def _append_label_events(
    *,
    cfg: Any,
    cfg_path: Path,
    records_df: pd.DataFrame,
    labels_effective: pd.DataFrame,
    round_index: int,
) -> None:
    if labels_effective.empty:
        return
    seq_map = records_df.set_index("id")["sequence"].to_dict() if "sequence" in records_df.columns else {}
    event_ids = labels_effective["id"].astype(str).tolist()
    events = build_label_events(
        ids=event_ids,
        sequences=[seq_map.get(label_id) for label_id in event_ids],
        y_obs=labels_effective["y"].tolist(),
        observed_round=int(round_index),
        src="ingest_y",
        note=None,
    )
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    LedgerWriter(ws).append_label(events)
