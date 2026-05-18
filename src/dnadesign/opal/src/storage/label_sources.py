"""
Shared and campaign-local training label sources for OPAL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from ..config.types import LabelSourceCampaignHistory, LabelSourceUSRSidecar, LocationUSR, RootConfig
from ..core.utils import OpalError
from .data_access import RecordsStore
from .locks import PathLock


def _coerce_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            value = json.loads(stripped)
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except Exception as exc:
        raise OpalError(f"observed label y_obs must be numeric vector-like: {exc}") from exc
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise OpalError("observed label y_obs must be finite and non-empty.")
    return arr.tolist()


def _known_id_set(known_ids: Iterable[str] | None) -> set[str] | None:
    if known_ids is None:
        return None
    return {str(v) for v in known_ids}


class TrainingLabelSource(Protocol):
    kind: str

    def validate(self, df: pd.DataFrame) -> None: ...

    def training_labels(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame: ...

    def labeled_id_set_leq_round(self, df: pd.DataFrame, as_of_round: int) -> set[str]: ...

    def labeled_id_set_any_round(self, df: pd.DataFrame) -> set[str]: ...


@dataclass(frozen=True)
class ObservedLabelStore:
    path: Path
    y_space: str
    id_column: str = "id"
    round_column: str = "observed_round"
    batch_column: str = "batch_id"
    dedup_policy: str = "latest_by_round"

    @property
    def kind(self) -> str:
        return "usr_sidecar"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise OpalError(f"Observed label source not found: {self.path}")
        try:
            return pd.read_parquet(self.path)
        except Exception as exc:
            raise OpalError(f"Failed to read observed label source {self.path}: {exc}") from exc

    def _load_for_append(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                columns=[
                    self.id_column,
                    self.round_column,
                    self.batch_column,
                    "y_space",
                    "y_obs",
                    "src",
                    "ts",
                ]
            )
        return self.load()

    def append_labels(
        self,
        labels: pd.DataFrame,
        *,
        observed_round: int,
        batch_id: str,
        src: str,
        if_exists: str,
        known_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if "id" not in labels.columns or "y" not in labels.columns:
            raise OpalError("Observed label append requires labels with columns ['id', 'y'].")
        if not str(self.y_space).strip():
            raise OpalError("Observed label append requires non-empty y_space.")

        known = _known_id_set(known_ids)
        incoming = labels.loc[:, ["id", "y"]].copy()
        incoming["id"] = incoming["id"].astype(str)
        if known is not None:
            unknown = sorted(set(incoming["id"]) - known)
            if unknown:
                raise OpalError(f"Observed labels contain unknown ids (sample={unknown[:10]}).")

        policy = str(if_exists or "fail").strip().lower()
        if policy not in {"fail", "skip", "replace"}:
            raise OpalError("--if-exists must be one of: fail, skip, replace.")

        with PathLock(self.path, lock_name="Observed label source"):
            existing = self._load_for_append()
            for col in [self.id_column, self.round_column, self.batch_column, "y_space", "y_obs", "src", "ts"]:
                if col not in existing.columns:
                    existing[col] = None
            existing[self.id_column] = existing[self.id_column].astype(str)

            existing_mask = (
                (existing[self.id_column].astype(str).isin(set(incoming["id"])))
                & (pd.to_numeric(existing[self.round_column], errors="coerce") == int(observed_round))
                & (existing["y_space"].astype(str) == str(self.y_space))
            )
            existing_ids = set(existing.loc[existing_mask, self.id_column].astype(str).tolist())
            if existing_ids:
                if policy == "fail":
                    raise OpalError(
                        f"Observed label source already has labels for round {int(observed_round)} "
                        f"(sample={sorted(existing_ids)[:10]})."
                    )
                if policy == "skip":
                    incoming = incoming.loc[~incoming["id"].isin(existing_ids)].copy()
                elif policy == "replace":
                    existing = existing.loc[~existing_mask].copy()

            if incoming.empty:
                return pd.DataFrame(columns=["id", "y"])

            ts = pd.Timestamp.now("UTC").isoformat()
            rows = pd.DataFrame(
                {
                    self.id_column: incoming["id"].astype(str).tolist(),
                    self.round_column: [int(observed_round)] * len(incoming),
                    self.batch_column: [str(batch_id)] * len(incoming),
                    "y_space": [str(self.y_space)] * len(incoming),
                    "y_obs": [_coerce_float_list(v) for v in incoming["y"].tolist()],
                    "src": [str(src)] * len(incoming),
                    "ts": [ts] * len(incoming),
                }
            )
            out = pd.concat([existing, rows], ignore_index=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.tmp")
            out.to_parquet(tmp, index=False)
            tmp.replace(self.path)
        return pd.DataFrame({"id": rows[self.id_column].astype(str).tolist(), "y": rows["y_obs"].tolist()})

    def _validated_frame(self, *, known_ids: Iterable[str] | None = None) -> pd.DataFrame:
        df = self.load()
        required = [self.id_column, self.round_column, self.batch_column, "y_space", "y_obs"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise OpalError(f"Observed label source missing required columns: {missing}")

        out = df.copy()
        out[self.id_column] = out[self.id_column].astype(str)
        out["y_space"] = out["y_space"].astype(str)
        out = out.loc[out["y_space"] == str(self.y_space)].copy()

        known = _known_id_set(known_ids)
        if known is not None:
            unknown = sorted(set(out[self.id_column].astype(str)) - known)
            if unknown:
                raise OpalError(f"Observed label source contains unknown ids (sample={unknown[:10]}).")

        try:
            out[self.round_column] = out[self.round_column].astype(int)
        except Exception as exc:
            raise OpalError(f"Observed label source round column '{self.round_column}' must be integer-like.") from exc

        out["y"] = out["y_obs"].map(_coerce_float_list)
        out["_row_order"] = np.arange(len(out), dtype=int)

        duplicate_keys = out.duplicated(subset=[self.id_column, self.round_column], keep=False)
        if duplicate_keys.any():
            policy = str(self.dedup_policy or "latest_by_round").strip().lower()
            if policy == "error_on_duplicate":
                sample = (
                    out.loc[duplicate_keys, [self.id_column, self.round_column]]
                    .drop_duplicates()
                    .head(10)
                    .to_dict(orient="records")
                )
                raise OpalError(f"Observed label source has duplicate id/round labels (sample={sample}).")
            if policy == "latest_by_round":
                out = out.sort_values("_row_order").groupby([self.id_column, self.round_column], as_index=False).tail(1)
            elif policy == "all_events":
                pass
            else:
                raise OpalError(
                    f"Unknown observed label dedup_policy: {self.dedup_policy!r} "
                    "(expected: latest_by_round | all_events | error_on_duplicate)."
                )

        return out

    def training_labels(
        self,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
        known_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        frame = self._validated_frame(known_ids=known_ids)
        if cumulative_training:
            frame = frame.loc[frame[self.round_column] <= int(as_of_round)].copy()
        else:
            frame = frame.loc[frame[self.round_column] == int(as_of_round)].copy()
        if frame.empty:
            return pd.DataFrame(columns=["id", "y", "r"])

        policy = str(dedup_policy or "latest_only").strip().lower()
        if policy not in {"latest_only", "all_rounds", "error_on_duplicate"}:
            raise OpalError(
                f"Unknown label_cross_round_deduplication_policy: {dedup_policy!r} "
                "(expected: latest_only | all_rounds | error_on_duplicate)."
            )

        if policy == "latest_only":
            selected = (
                frame.sort_values([self.id_column, self.round_column, "_row_order"])
                .groupby(self.id_column, as_index=False)
                .tail(1)
                .copy()
            )
        elif policy == "all_rounds":
            selected = frame.copy()
        else:
            duplicated_ids = frame[self.id_column].duplicated(keep=False)
            if duplicated_ids.any():
                sample = sorted(frame.loc[duplicated_ids, self.id_column].astype(str).unique().tolist())[:10]
                raise OpalError(f"Duplicate labels for ids at multiple rounds (sample={sample}).")
            selected = frame.copy()

        selected = selected.sort_values([self.id_column, self.round_column, "_row_order"])
        return pd.DataFrame(
            {
                "id": selected[self.id_column].astype(str).tolist(),
                "y": selected["y"].tolist(),
                "r": selected[self.round_column].astype(int).tolist(),
            }
        )

    def observed_ids(
        self,
        *,
        as_of_round: int | None = None,
        known_ids: Iterable[str] | None = None,
    ) -> set[str]:
        frame = self._validated_frame(known_ids=known_ids)
        if as_of_round is not None:
            frame = frame.loc[frame[self.round_column] <= int(as_of_round)]
        return set(frame[self.id_column].astype(str).tolist())


@dataclass(frozen=True)
class CampaignHistoryLabelSource:
    store: RecordsStore
    kind: str = "campaign_history"

    def validate(self, df: pd.DataFrame) -> None:
        self.store.validate_label_hist(df, require=True)

    def training_labels(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        return self.store.training_labels_with_round(
            df,
            int(as_of_round),
            cumulative_training=cumulative_training,
            dedup_policy=dedup_policy,
        )

    def labeled_id_set_leq_round(self, df: pd.DataFrame, as_of_round: int) -> set[str]:
        return self.store.labeled_id_set_leq_round(df, int(as_of_round))

    def labeled_id_set_any_round(self, df: pd.DataFrame) -> set[str]:
        return self.store.labeled_id_set_any_round(df)


@dataclass(frozen=True)
class SharedObservedLabelSource:
    store: ObservedLabelStore
    kind: str = "usr_sidecar"

    @staticmethod
    def _ids(df: pd.DataFrame) -> set[str]:
        if "id" not in df.columns:
            raise OpalError("records.parquet is missing required column 'id'.")
        return set(df["id"].astype(str).tolist())

    def validate(self, df: pd.DataFrame) -> None:
        _ = self.store._validated_frame(known_ids=self._ids(df))

    def training_labels(
        self,
        df: pd.DataFrame,
        as_of_round: int,
        *,
        cumulative_training: bool,
        dedup_policy: str,
    ) -> pd.DataFrame:
        return self.store.training_labels(
            int(as_of_round),
            cumulative_training=cumulative_training,
            dedup_policy=dedup_policy,
            known_ids=self._ids(df),
        )

    def labeled_id_set_leq_round(self, df: pd.DataFrame, as_of_round: int) -> set[str]:
        return self.store.observed_ids(as_of_round=int(as_of_round), known_ids=self._ids(df))

    def labeled_id_set_any_round(self, df: pd.DataFrame) -> set[str]:
        return self.store.observed_ids(known_ids=self._ids(df))


def label_source_from_config(cfg: RootConfig, store: RecordsStore) -> TrainingLabelSource:
    source_cfg = cfg.labels.source
    if isinstance(source_cfg, LabelSourceCampaignHistory):
        return CampaignHistoryLabelSource(store=store)
    if isinstance(source_cfg, LabelSourceUSRSidecar):
        loc = cfg.data.location
        if not isinstance(loc, LocationUSR):
            raise OpalError("labels.source.kind=usr_sidecar requires data.location.kind=usr.")
        if source_cfg.dataset != loc.dataset:
            raise OpalError("labels.source.dataset must match data.location.dataset for usr_sidecar labels.")
        sidecar_path = Path(loc.path) / source_cfg.dataset / source_cfg.path
        return SharedObservedLabelSource(
            store=ObservedLabelStore(
                path=sidecar_path,
                y_space=str(cfg.labels.y_space or ""),
                id_column=cfg.labels.id_column,
                round_column=cfg.labels.round_column,
                batch_column=cfg.labels.batch_column,
                dedup_policy=cfg.labels.dedup_policy,
            )
        )
    raise OpalError(f"Unsupported label source kind: {getattr(source_cfg, 'kind', None)!r}")


def _counts_by_round(frame: pd.DataFrame, round_column: str) -> dict[int, int]:
    if frame.empty:
        return {}
    counts = frame.groupby(round_column).size().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


def _validate_expected_length(frame: pd.DataFrame, expected_length: int | None, *, id_column: str) -> None:
    if expected_length is None or frame.empty:
        return
    bad = frame.loc[frame["y"].map(len) != int(expected_length)]
    if bad.empty:
        return
    sample = bad[[id_column, "y"]].head(5).to_dict(orient="records")
    raise OpalError(
        "Observed label source y_obs length mismatch: "
        f"expected {int(expected_length)} values per label (sample={sample})."
    )


def label_source_status(
    cfg: RootConfig,
    store: RecordsStore,
    df: pd.DataFrame,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Return a small machine-readable summary of the configured label source."""

    source = label_source_from_config(cfg, store)
    if isinstance(source, CampaignHistoryLabelSource):
        hist_col = store.label_hist_col()
        out: dict[str, Any] = {
            "kind": "campaign_history",
            "column": hist_col,
            "exists": hist_col in df.columns,
            "valid": None,
            "label_count": 0,
            "available_rounds": [],
            "counts_by_round": {},
        }
        if hist_col not in df.columns:
            return out
        bad: list[dict[str, str]] = []
        counts: dict[int, int] = {}
        for row_id, cell in df[["id", hist_col]].itertuples(index=False, name=None):
            try:
                entries = store._parse_hist_cell_strict(cell)
            except Exception as exc:
                bad.append({"id": str(row_id), "error": str(exc)})
                if len(bad) >= 5:
                    break
                continue
            for entry in entries:
                if entry.get("kind") != "label":
                    continue
                try:
                    observed_round = int(entry.get("observed_round"))
                except Exception:
                    continue
                counts[observed_round] = counts.get(observed_round, 0) + 1
        if bad:
            error = f"label_hist validation failed (sample={bad})."
            if strict:
                raise OpalError(error)
            out.update({"valid": False, "error": error})
            return out
        out.update(
            {
                "valid": True,
                "label_count": int(sum(counts.values())),
                "available_rounds": sorted(counts),
                "counts_by_round": counts,
            }
        )
        return out

    if isinstance(source, SharedObservedLabelSource):
        obs = source.store
        out = {
            "kind": "usr_sidecar",
            "path": str(obs.path),
            "y_space": obs.y_space,
            "id_column": obs.id_column,
            "round_column": obs.round_column,
            "batch_column": obs.batch_column,
            "dedup_policy": obs.dedup_policy,
            "exists": obs.path.exists(),
            "valid": None,
            "label_count": 0,
            "available_rounds": [],
            "counts_by_round": {},
        }
        if not obs.path.exists():
            if strict:
                raise OpalError(f"Observed label source not found: {obs.path}")
            return out
        try:
            frame = obs._validated_frame(known_ids=source._ids(df))
            _validate_expected_length(frame, cfg.data.y_expected_length, id_column=obs.id_column)
        except OpalError as exc:
            if strict:
                raise
            out.update({"valid": False, "error": str(exc)})
            return out

        counts = _counts_by_round(frame, obs.round_column)
        out.update(
            {
                "valid": True,
                "label_count": int(len(frame)),
                "available_rounds": sorted(counts),
                "counts_by_round": counts,
            }
        )
        return out

    raise OpalError(f"Unsupported label source kind: {getattr(source, 'kind', None)!r}")
