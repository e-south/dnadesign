"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/label_sources.py

Shared and campaign-local training label sources for OPAL.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from ..config.types import LabelSourceCampaignHistory, LabelSourceUSRSidecar, LocationUSR, RootConfig
from ..core.leakage import (
    assert_no_leakage_violations,
    build_shared_label_source_contamination_report,
)
from ..core.utils import OpalError
from .candidate_exclusion_projection import candidate_exclusion_sets_from_config
from .data_access import RecordsStore
from .locks import PathLock
from .observed_label_promotion import (
    ObservedLabelPromotionBinding,
    VerifiedObservedLabelPromotion,
    verify_observed_label_promotion,
)


def _coerce_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            value = json.loads(stripped)
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        arr = np.asarray(value, dtype=float)
    except Exception as exc:
        raise OpalError(f"observed label y_obs must be numeric vector-like: {exc}") from exc
    if arr.ndim != 1:
        raise OpalError("observed label y_obs must be a one-dimensional numeric vector.")
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

    def observed_events(self, df: pd.DataFrame, as_of_round: int, *, y_space: str | None) -> pd.DataFrame: ...

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
    promotion: ObservedLabelPromotionBinding | None = None

    @property
    def kind(self) -> str:
        return "usr_sidecar"

    @property
    def manifest_pinned(self) -> bool:
        return self.promotion is not None

    def verified_promotion(self) -> VerifiedObservedLabelPromotion | None:
        if self.promotion is None:
            return None
        verified = verify_observed_label_promotion(self.promotion)
        if self.path.resolve() != verified.label_path:
            raise OpalError(
                "Manifest-pinned observed-label store path does not match the verified label artifact: "
                f"store={self.path.resolve()}, manifest={verified.label_path}."
            )
        return verified

    def load(self) -> pd.DataFrame:
        _ = self.verified_promotion()
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
        if self.manifest_pinned:
            raise OpalError(
                "Observed label source is manifest-pinned and immutable; generic ingest-y cannot modify it. "
                "Publish the Parquet artifact and promotion manifest through the owning study workflow."
            )
        if "id" not in labels.columns or "y" not in labels.columns:
            raise OpalError("Observed label append requires labels with columns ['id', 'y'].")
        if not str(self.y_space).strip():
            raise OpalError("Observed label append requires non-empty y_space.")
        if batch_id is None or not isinstance(batch_id, str) or not batch_id.strip() or batch_id != batch_id.strip():
            raise OpalError("Observed label append requires batch_id to be a canonical, non-null, non-blank string.")

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
                    self.batch_column: [batch_id] * len(incoming),
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
        if "display_label" not in out.columns:
            out["display_label"] = pd.Series([None] * len(out), index=out.index, dtype=object)
        else:
            display_labels: list[str | None] = []
            invalid_display_labels: list[bool] = []
            for value in out["display_label"].tolist():
                missing = pd.isna(value)
                is_missing = isinstance(missing, (bool, np.bool_)) and bool(missing)
                valid = is_missing or (isinstance(value, str) and bool(value) and value == value.strip())
                display_labels.append(None if is_missing else value)
                invalid_display_labels.append(not valid)
            invalid_display_label_mask = pd.Series(invalid_display_labels, index=out.index)
            if invalid_display_label_mask.any():
                sample = (
                    out.loc[invalid_display_label_mask, [self.id_column, self.round_column, "display_label"]]
                    .head(10)
                    .to_dict(orient="records")
                )
                raise OpalError(
                    "Observed label source display_label values must be null or canonical non-blank strings "
                    f"(sample={sample})."
                )
            out["display_label"] = pd.Series(display_labels, index=out.index, dtype=object)
        raw_batch_ids = out[self.batch_column]
        batch_id_strings = raw_batch_ids.map(lambda value: None if pd.isna(value) else str(value))
        canonical_batch_ids = batch_id_strings.map(lambda value: None if pd.isna(value) else str(value).strip())
        invalid_batch_ids = (
            canonical_batch_ids.isna() | canonical_batch_ids.eq("") | batch_id_strings.ne(canonical_batch_ids)
        )
        if invalid_batch_ids.any():
            sample = out.loc[invalid_batch_ids, [self.id_column, self.round_column]].head(10).to_dict(orient="records")
            raise OpalError(
                f"Observed label source batch column '{self.batch_column}' must contain canonical, non-null, "
                f"non-blank identifiers (sample={sample})."
            )
        out[self.batch_column] = batch_id_strings.astype(str)
        out["y_space"] = out["y_space"].astype(str)
        expected_y_space = str(self.y_space)
        observed_y_spaces = sorted(out["y_space"].unique().tolist())
        if observed_y_spaces != [expected_y_space]:
            raise OpalError(
                "Observed label source must contain one Y space matching the configured campaign; "
                f"expected={expected_y_space!r}, observed={observed_y_spaces!r}."
            )

        known = _known_id_set(known_ids)
        if known is not None:
            unknown = sorted(set(out[self.id_column].astype(str)) - known)
            if unknown:
                raise OpalError(f"Observed label source contains unknown ids (sample={unknown[:10]}).")

        raw_rounds = out[self.round_column]
        numeric_rounds = pd.to_numeric(raw_rounds, errors="coerce")
        round_values = numeric_rounds.to_numpy(dtype=float)
        has_boolean = bool(raw_rounds.map(lambda value: isinstance(value, (bool, np.bool_))).any())
        invalid_round = (
            has_boolean
            or not np.all(np.isfinite(round_values))
            or bool(np.any(round_values < 0))
            or bool(np.any(round_values != np.floor(round_values)))
        )
        if invalid_round:
            raise OpalError(
                f"Observed label source round column '{self.round_column}' must contain nonnegative integers."
            )
        out[self.round_column] = numeric_rounds.astype(int)

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

    def observed_events(
        self,
        as_of_round: int,
        *,
        known_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return every verified source event visible to a run, before cross-round deduplication."""

        frame = self._validated_frame(known_ids=known_ids)
        frame = frame.loc[frame[self.round_column] <= int(as_of_round)].copy()
        frame = frame.sort_values([self.id_column, self.round_column, "_row_order"], kind="stable")
        return pd.DataFrame(
            {
                "id": frame[self.id_column].astype(str).tolist(),
                "display_label": frame["display_label"].tolist(),
                "observed_round": frame[self.round_column].astype(int).tolist(),
                "batch_id": frame[self.batch_column].astype(str).tolist(),
                "y_space": frame["y_space"].astype(str).tolist(),
                "y_obs": frame["y"].tolist(),
                "label_source_kind": [self.kind] * len(frame),
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

    def observed_events(self, df: pd.DataFrame, as_of_round: int, *, y_space: str | None) -> pd.DataFrame:
        events = self.store.training_labels_with_round(
            df,
            int(as_of_round),
            cumulative_training=True,
            dedup_policy="all_rounds",
        ).sort_values(["id", "r"], kind="stable")
        resolved_y_space = None if y_space is None or not str(y_space).strip() else str(y_space).strip()
        return pd.DataFrame(
            {
                "id": events["id"].astype(str).tolist(),
                "display_label": [None] * len(events),
                "observed_round": events["r"].astype(int).tolist(),
                "batch_id": [None] * len(events),
                "y_space": [resolved_y_space] * len(events),
                "y_obs": events["y"].tolist(),
                "label_source_kind": [self.kind] * len(events),
            }
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

    def observed_events(self, df: pd.DataFrame, as_of_round: int, *, y_space: str | None) -> pd.DataFrame:
        configured_y_space = "" if y_space is None else str(y_space).strip()
        if configured_y_space != str(self.store.y_space):
            raise OpalError(
                "Configured label Y space does not match the shared observed-label store: "
                f"configured={configured_y_space!r}, store={str(self.store.y_space)!r}."
            )
        return self.store.observed_events(
            int(as_of_round),
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
        dataset_root = Path(loc.path) / source_cfg.dataset
        sidecar_path = dataset_root / source_cfg.path
        promotion = None
        if source_cfg.manifest_path is not None:
            study_id = str(cfg.ownership.study_id or "").strip()
            if not study_id:
                raise OpalError("Manifest-pinned observed labels require non-empty ownership.study_id.")
            promotion = ObservedLabelPromotionBinding(
                dataset_root=dataset_root,
                manifest_path=source_cfg.manifest_path,
                label_path=source_cfg.path,
                campaign_slug=cfg.campaign.slug,
                study_id=study_id,
                y_space=str(cfg.labels.y_space or ""),
                candidate_path="records.parquet",
                candidate_id_column=cfg.labels.id_column,
                candidate_x_column=cfg.data.x_column_name,
                candidate_exclusion_sets=candidate_exclusion_sets_from_config(cfg),
            )
        return SharedObservedLabelSource(
            store=ObservedLabelStore(
                path=sidecar_path,
                y_space=str(cfg.labels.y_space or ""),
                id_column=cfg.labels.id_column,
                round_column=cfg.labels.round_column,
                batch_column=cfg.labels.batch_column,
                dedup_policy=cfg.labels.dedup_policy,
                promotion=promotion,
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
        leakage_report = build_shared_label_source_contamination_report(cfg=cfg, store=store, df=df)
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
            "leakage": leakage_report.to_dict(),
            "manifest_pinned": obs.manifest_pinned,
            "mutable": not obs.manifest_pinned,
        }
        if obs.promotion is not None:
            manifest_path = Path(obs.promotion.dataset_root) / obs.promotion.manifest_path
            out.update(
                {
                    "manifest_path": str(manifest_path),
                    "manifest_exists": manifest_path.exists(),
                    "artifact_exists": obs.path.exists(),
                }
            )
        if leakage_report.violations:
            try:
                assert_no_leakage_violations(leakage_report)
            except OpalError as exc:
                if strict:
                    raise
                out.update({"valid": False, "error": str(exc)})
                return out
        verified: VerifiedObservedLabelPromotion | None = None
        if obs.manifest_pinned:
            try:
                verified = obs.verified_promotion()
            except OpalError as exc:
                if strict:
                    raise
                out.update({"valid": False, "error": str(exc)})
                return out
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
        verified = verified or obs.verified_promotion()
        out.update(
            {
                "valid": True,
                "label_count": int(len(frame)),
                "available_rounds": sorted(counts),
                "counts_by_round": counts,
            }
        )
        if verified is not None:
            out.update(
                {
                    "manifest_sha256": verified.manifest_sha256,
                    "label_sha256": verified.label_sha256,
                    "promoted_row_count": verified.row_count,
                    "study_provenance_schema_id": verified.study_provenance_schema_id,
                    "study_provenance_path": str(verified.study_provenance_path),
                    "study_provenance_sha256": verified.study_provenance_sha256,
                    "candidate_exclusion_set_id": verified.candidate_exclusion_set_id,
                    "candidate_exclusion_entries_sha256": verified.candidate_exclusion_entries_sha256,
                    "candidate_exclusion_entry_count": verified.candidate_exclusion_entry_count,
                }
            )
        return out

    raise OpalError(f"Unsupported label source kind: {getattr(source, 'kind', None)!r}")
