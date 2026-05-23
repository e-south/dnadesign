"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/sources/input_rows.py

USR input-row selection and label contracts for Construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Protocol

from ..contracts.config import JobConfig
from ..contracts.errors import ValidationError


class _BatchLike(Protocol):
    num_rows: int

    def to_pydict(self) -> dict[str, list[object]]: ...


class _SchemaLike(Protocol):
    names: Iterable[str]


class USRRowSource(Protocol):
    def scan(self, *, columns: List[str], include_overlays: bool) -> Iterable[_BatchLike]: ...

    def schema(self) -> _SchemaLike: ...


def scan_usr_rows(ds: USRRowSource, *, columns: List[str], ids: List[str] | None) -> List[dict[str, object]]:
    wanted = [str(value) for value in (ids or []) if str(value).strip()]
    wanted_set = set(wanted) if wanted else None
    found: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []

    for batch in ds.scan(columns=columns, include_overlays=True):
        payload = batch.to_pydict()
        row_count = batch.num_rows
        for idx in range(row_count):
            row = {name: payload[name][idx] for name in payload}
            row_id = str(row["id"])
            if wanted_set is not None:
                if row_id not in wanted_set:
                    continue
                found[row_id] = row
            else:
                ordered.append(row)

    if wanted_set is not None:
        missing = [row_id for row_id in wanted if row_id not in found]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValidationError(f"{len(missing)} requested input id(s) were not found. Sample: {preview}.")
        return [found[row_id] for row_id in wanted]
    return ordered


def input_fields(cfg: JobConfig) -> List[str]:
    fields = {"id"}
    if cfg.job.input.field is not None:
        fields.add(cfg.job.input.field)
    for part in cfg.job.parts:
        if part.sequence.source == "input_field":
            fields.add(str(part.sequence.field))
    return sorted(fields)


def classic_input_scan_fields(ds: USRRowSource, cfg: JobConfig) -> List[str]:
    fields = set(input_fields(cfg))
    available = set(ds.schema().names)
    if "usr_label__primary" in available:
        fields.add("usr_label__primary")
    if "usr_label__aliases" in available:
        fields.add("usr_label__aliases")
    return sorted(fields)


def normalize_input_scan_fields(ds: USRRowSource, cfg: JobConfig) -> List[str]:
    return normalize_input_scan_fields_for_schema(
        input_field=cfg.job.input.field,
        available_fields=ds.schema().names,
    )


def normalize_input_scan_fields_for_schema(*, input_field: str | None, available_fields: Iterable[str]) -> List[str]:
    if input_field is None:
        raise ValidationError("job.input.field is required when job.mode='normalize_anchor'.")
    fields = {"id", input_field}
    available = set(available_fields)
    for field_name in (
        "usr_label__primary",
        "usr_label__aliases",
        "seq_annot__features",
        "seq_annot__record_id",
        "seq_annot__record_name",
    ):
        if field_name in available:
            fields.add(field_name)
    return sorted(fields)


def input_usr_labels(row: dict[str, object]) -> tuple[str | None, List[str]]:
    primary_raw = row.get("usr_label__primary")
    primary = str(primary_raw).strip() if primary_raw is not None and str(primary_raw).strip() else None

    aliases_raw = row.get("usr_label__aliases")
    aliases: list[str] = []
    if isinstance(aliases_raw, list):
        raw_values = aliases_raw
    elif aliases_raw is None:
        raw_values = []
    else:
        raw_values = [aliases_raw]
    for value in raw_values:
        text = str(value or "").strip()
        if not text or text == primary or text in aliases:
            continue
        aliases.append(text)
    return primary, aliases


def require_distinct_input_output_or_opt_in(
    *,
    cfg: JobConfig,
    input_root: Path,
    output_root: Path,
) -> None:
    if (
        input_root == output_root
        and cfg.job.input.source.dataset == cfg.job.output.target.dataset
        and not cfg.job.output.allow_same_as_input
    ):
        raise ValidationError(
            "Output dataset resolves to the same root/dataset as input. "
            "Set output.allow_same_as_input=true only when recursive accumulation is intentional."
        )
