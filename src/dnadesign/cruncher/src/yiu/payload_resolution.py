"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/payload_resolution.py

Resolve YIU payload sequences from direct input or Sample hit artifacts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.cruncher.bio import normalize_iupac
from dnadesign.cruncher.yiu.errors import (
    YIU_PATH_INVALID,
    YIU_SAMPLE_HIT_AMBIGUOUS,
    YIU_SAMPLE_HIT_SEQUENCE_MISSING,
    YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE,
    raise_yiu_error,
)
from dnadesign.cruncher.yiu.spec_models import InputSpec, SampleHitInput


@dataclass(frozen=True)
class ResolvedInputPayload:
    input_kind: str
    payload_sequence: str
    payload_label: str | None
    site_label: str | None
    provenance: dict[str, object]
    hit_row: dict[str, Any] | None
    source_artifact_path: Path | None
    sample_workspace_root: Path | None


def _metadata_text(sample_hit: SampleHitInput, key: str) -> str | None:
    raw = sample_hit.metadata.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _normalize_sequence(value: str, *, ctx: str) -> str:
    try:
        return normalize_iupac(value)
    except Exception as exc:
        raise_yiu_error(YIU_SAMPLE_HIT_SEQUENCE_MISSING, f"{ctx} is not a normalized DNA sequence ({exc})")


def _resolve_workspace_ref(raw: str, *, workspace_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise_yiu_error(YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE, f"sample-hit source workspace not found: {resolved}")

    candidates: list[Path] = []
    for candidate_root in (workspace_root, workspace_root.parent):
        candidate = (candidate_root / path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise_yiu_error(
        YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE,
        "sample-hit source workspace not found; use an absolute path or a sibling workspace path/name "
        f"that resolves from the current workspace root or its parent ({searched})",
    )


def _resolve_source_artifact_path(
    sample_hit: SampleHitInput, *, workspace_root: Path
) -> tuple[Path | None, Path | None]:
    if sample_hit.source_artifact_path is not None:
        raw_path = Path(sample_hit.source_artifact_path).expanduser()
        if raw_path.is_absolute():
            artifact_path = raw_path.resolve()
        else:
            if any(part == ".." for part in raw_path.parts):
                raise_yiu_error(
                    YIU_PATH_INVALID,
                    "input.sample_hit.source_artifact_path must not traverse outside the current workspace",
                )
            artifact_path = (workspace_root / raw_path).resolve()
        return artifact_path, _infer_sample_workspace_root(artifact_path)

    source_workspace = _metadata_text(sample_hit, "source_workspace")
    source_artifact = sample_hit.source_artifact or _metadata_text(sample_hit, "source_artifact")
    if source_workspace is None or source_artifact is None:
        return None, None
    workspace_ref = _resolve_workspace_ref(source_workspace, workspace_root=workspace_root)
    artifact_path = Path(source_artifact).expanduser()
    artifact_path = (
        artifact_path.resolve() if artifact_path.is_absolute() else (workspace_ref / artifact_path).resolve()
    )
    return artifact_path, workspace_ref.resolve()


def _infer_sample_workspace_root(artifact_path: Path) -> Path | None:
    resolved = artifact_path.resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if parent.name == "outputs":
            return parent.parent.resolve()
    return None


def _resolve_hit_table_fields(columns: set[str], artifact_path: Path) -> tuple[str, str]:
    if {"elite_id", "elite_sequence"}.issubset(columns):
        return "elite_id", "elite_sequence"
    if {"hit_id", "payload_sequence"}.issubset(columns):
        return "hit_id", "payload_sequence"
    if {"id", "sequence"}.issubset(columns):
        return "id", "sequence"
    raise_yiu_error(
        YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE,
        f"sample-hit source artifact does not expose a supported public hit table: {artifact_path.name}",
    )


def _load_csv_hit_rows(artifact_path: Path, *, sample_hit: SampleHitInput) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    with artifact_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise_yiu_error(YIU_SAMPLE_HIT_SEQUENCE_MISSING, f"sample-hit source artifact is empty: {artifact_path}")
        id_field, _sequence_field = _resolve_hit_table_fields(set(reader.fieldnames), artifact_path)
        for row in reader:
            if str(row.get(id_field, "")).strip() == sample_hit.hit_id:
                matches.append(dict(row))
    return matches


def _load_parquet_hit_rows(artifact_path: Path, *, sample_hit: SampleHitInput) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise_yiu_error(YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE, f"parquet resolution requires pandas ({exc})")

    columns: set[str]
    try:
        import pyarrow.parquet as pq  # type: ignore

        columns = set(pq.read_schema(artifact_path).names)
    except Exception:
        columns = set(pd.read_parquet(artifact_path, nrows=0).columns)
    id_field, sequence_field = _resolve_hit_table_fields(columns, artifact_path)
    projected_columns = [id_field, sequence_field]
    if "per_tf_json" in columns:
        projected_columns.append("per_tf_json")
    try:
        frame = pd.read_parquet(artifact_path, columns=projected_columns, filters=[(id_field, "==", sample_hit.hit_id)])
    except Exception:
        frame = pd.read_parquet(artifact_path, columns=projected_columns)
        frame = frame.loc[frame[id_field].astype(str) == sample_hit.hit_id]
    return frame.to_dict(orient="records")


def _load_matching_hit_rows(artifact_path: Path, *, sample_hit: SampleHitInput) -> list[dict[str, Any]]:
    if not artifact_path.exists():
        raise_yiu_error(YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE, f"sample-hit source artifact not found: {artifact_path}")
    suffix = artifact_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_hit_rows(artifact_path, sample_hit=sample_hit)
    if suffix == ".parquet":
        return _load_parquet_hit_rows(artifact_path, sample_hit=sample_hit)
    raise_yiu_error(
        YIU_SAMPLE_HIT_UNSUPPORTED_SOURCE,
        f"unsupported sample-hit source artifact: {artifact_path.name}",
    )


def _resolve_sample_hit_payload(
    sample_hit: SampleHitInput,
    *,
    workspace_root: Path,
) -> tuple[str, dict[str, Any] | None, Path | None, Path | None]:
    artifact_path, sample_workspace_root = _resolve_source_artifact_path(sample_hit, workspace_root=workspace_root)
    direct_sequence = sample_hit.payload_sequence
    selected_row: dict[str, Any] | None = None
    derived_sequence: str | None = None
    if artifact_path is not None:
        rows = _load_matching_hit_rows(artifact_path, sample_hit=sample_hit)
        if not rows:
            raise_yiu_error(
                YIU_SAMPLE_HIT_SEQUENCE_MISSING,
                f"sample-hit hit_id={sample_hit.hit_id!r} was not found in {artifact_path.name}",
            )
        id_field, sequence_field = _resolve_hit_table_fields(set(rows[0].keys()), artifact_path)
        sequences = sorted(
            {
                _normalize_sequence(
                    str(row.get(sequence_field, "")).strip(), ctx=f"{artifact_path.name}:{sequence_field}"
                )
                for row in rows
                if str(row.get(sequence_field, "")).strip()
            }
        )
        if not sequences:
            raise_yiu_error(
                YIU_SAMPLE_HIT_SEQUENCE_MISSING,
                f"sample-hit hit_id={sample_hit.hit_id!r} is missing a payload sequence in {artifact_path.name}",
            )
        if len(sequences) != 1:
            raise_yiu_error(
                YIU_SAMPLE_HIT_AMBIGUOUS,
                "sample-hit lookup resolved to multiple payload sequences: " + ", ".join(sequences),
            )
        derived_sequence = sequences[0]
        selected_row = next(
            (
                dict(row)
                for row in rows
                if _normalize_sequence(
                    str(row.get(sequence_field, "")).strip(), ctx=f"{artifact_path.name}:{sequence_field}"
                )
                == derived_sequence
                and str(row.get(id_field, "")).strip() == sample_hit.hit_id
            ),
            dict(rows[0]),
        )
    if direct_sequence is not None and derived_sequence is not None and direct_sequence != derived_sequence:
        raise_yiu_error(
            YIU_SAMPLE_HIT_AMBIGUOUS,
            "sample_hit.payload_sequence does not match the resolved public source artifact sequence",
        )
    if direct_sequence is not None:
        return direct_sequence, selected_row, artifact_path, sample_workspace_root
    if derived_sequence is not None:
        return derived_sequence, selected_row, artifact_path, sample_workspace_root
    raise_yiu_error(
        YIU_SAMPLE_HIT_SEQUENCE_MISSING,
        "sample_hit did not resolve a payload sequence; supply payload_sequence or a readable source artifact",
    )


def resolve_input_payload(input_spec: InputSpec, *, workspace_root: Path, spec_name: str) -> ResolvedInputPayload:
    if input_spec.kind == "user_sequence":
        assert input_spec.user_sequence is not None
        return ResolvedInputPayload(
            input_kind="user_sequence",
            payload_sequence=input_spec.user_sequence.sequence,
            payload_label=None,
            site_label=None,
            provenance={
                "spec_name": spec_name,
                "sample_name": None,
                "hit_id": None,
                "source_artifact_path": None,
            },
            hit_row=None,
            source_artifact_path=None,
            sample_workspace_root=None,
        )

    assert input_spec.sample_hit is not None
    payload_sequence, hit_row, artifact_path, sample_workspace_root = _resolve_sample_hit_payload(
        input_spec.sample_hit,
        workspace_root=workspace_root,
    )
    payload_label = (
        _metadata_text(input_spec.sample_hit, "payload_label")
        or _metadata_text(input_spec.sample_hit, "tf_name")
        or _metadata_text(input_spec.sample_hit, "motif_name")
    )
    site_label = _metadata_text(input_spec.sample_hit, "site_label")
    provenance = {
        "spec_name": spec_name,
        "sample_name": input_spec.sample_hit.sample_name,
        "hit_id": input_spec.sample_hit.hit_id,
        "source_artifact_path": None if artifact_path is None else str(artifact_path.resolve()),
        "source_workspace": None if sample_workspace_root is None else str(sample_workspace_root.resolve()),
        "metadata": dict(input_spec.sample_hit.metadata),
    }
    return ResolvedInputPayload(
        input_kind="sample_hit",
        payload_sequence=payload_sequence,
        payload_label=payload_label,
        site_label=site_label,
        provenance=provenance,
        hit_row=hit_row,
        source_artifact_path=artifact_path,
        sample_workspace_root=sample_workspace_root,
    )
