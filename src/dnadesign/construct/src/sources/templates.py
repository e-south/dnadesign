"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/sources/templates.py

Template source loading contracts for Construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.usr import Dataset

from ..contracts.config import JobConfig, NormalizeTemplateConfig
from ..contracts.errors import ValidationError
from ..realization.sequences import ensure_dna_text
from .input_rows import scan_usr_rows
from .paths import resolve_optional_path, resolve_usr_root


@dataclass(frozen=True)
class ResolvedTemplate:
    id: str
    kind: str
    sequence: str
    source: str
    dataset: str | None
    field: str | None
    record_id: str | None
    circular: bool


def load_template_sequence(base_dir: Path, cfg: JobConfig) -> ResolvedTemplate:
    template = cfg.job.template
    if template is None:
        raise ValidationError("job.template is required for classic construct jobs.")
    template_source = template.source
    if template_source.kind == "literal":
        seq = ensure_dna_text(template_source.sequence, label="template.source.sequence")
        return ResolvedTemplate(
            id=template.id,
            kind="literal",
            sequence=seq,
            source=template_source.label or "template.source.sequence",
            dataset=None,
            field=None,
            record_id=None,
            circular=bool(template.circular),
        )

    if template_source.kind == "path":
        return _load_path_template(
            base_dir=base_dir,
            path_value=template_source.path,
            label=template_source.label,
            sequence_label_prefix="template.source.path",
            template_id=template.id,
            default_id=None,
            circular=bool(template.circular),
            missing_message=f"Template path not found: {template_source.path}",
            unreadable_message="Template path must resolve to a readable file",
            empty_message_prefix="Template file is empty",
            fasta_message_prefix="Template FASTA",
        )

    if template_source.kind != "usr":
        raise ValidationError(f"Unsupported template.source.kind '{template_source.kind}'.")

    template_root = resolve_usr_root(
        base_dir,
        template_source.root or cfg.job.input.source.root,
        label="template.source.root or job.input.source.root",
    )
    template_ds = Dataset(template_root, str(template_source.dataset))
    if not template_ds.records_path.exists():
        raise ValidationError(f"Template dataset not initialized: {template_ds.records_path}")
    rows = scan_usr_rows(
        template_ds,
        columns=["id", str(template_source.field)],
        ids=[str(template_source.record_id)],
    )
    if len(rows) != 1:
        raise ValidationError(
            f"Template selection must resolve exactly one row in dataset '{template_source.dataset}'."
        )
    row = rows[0]
    raw = row.get(str(template_source.field))
    if raw is None:
        raise ValidationError(
            f"Template record '{template_source.record_id}' is missing field '{template_source.field}'."
        )
    seq = ensure_dna_text(
        str(raw),
        label=f"template field '{template_source.field}' in dataset '{template_source.dataset}'",
    )
    return ResolvedTemplate(
        id=template.id,
        kind="usr",
        sequence=seq,
        source=template_source.label or f"usr:{template_source.dataset}:{template_source.record_id}",
        dataset=str(template_source.dataset),
        field=str(template_source.field),
        record_id=str(template_source.record_id),
        circular=bool(template.circular),
    )


def load_normalize_template(*, base_dir: Path, cfg: NormalizeTemplateConfig) -> ResolvedTemplate:
    template_id = str(cfg.id or "").strip()
    source = cfg.source
    if source.kind == "literal":
        sequence = ensure_dna_text(source.sequence, label="normalize_anchor template.source.sequence")
        return ResolvedTemplate(
            id=template_id or source.label or "normalize_anchor_template",
            kind="literal",
            sequence=sequence,
            source=source.label or "normalize_anchor.template.source.sequence",
            dataset=None,
            field=None,
            record_id=None,
            circular=bool(cfg.circular),
        )
    if source.kind == "path":
        return _load_path_template(
            base_dir=base_dir,
            path_value=source.path,
            label=source.label,
            sequence_label_prefix="normalize_anchor.template.source.path",
            template_id=template_id,
            default_id=None,
            circular=bool(cfg.circular),
            missing_message=f"Normalize-anchor template path not found: {source.path}",
            unreadable_message="Normalize-anchor template path not found",
            empty_message_prefix="Normalize-anchor template file is empty",
            fasta_message_prefix="Normalize-anchor template FASTA files",
        )

    template_root = resolve_usr_root(base_dir, source.root, label="normalize_anchor.template.source.root")
    dataset = Dataset(template_root, source.dataset)
    if not dataset.records_path.exists():
        raise ValidationError(f"Normalize-anchor template dataset not initialized: {dataset.records_path}")
    rows = scan_usr_rows(
        dataset,
        columns=["id", str(source.field)],
        ids=[str(source.record_id)],
    )
    if len(rows) != 1:
        raise ValidationError(
            f"Normalize-anchor template record not found: {source.dataset}:{source.record_id} field={source.field}"
        )
    record = rows[0]
    raw = record.get(str(source.field))
    if raw is None:
        raise ValidationError(
            f"Normalize-anchor template record '{source.record_id}' is missing field '{source.field}'."
        )
    sequence = ensure_dna_text(str(raw), label=f"{source.dataset}:{source.record_id}:{source.field}")
    return ResolvedTemplate(
        id=template_id or source.record_id,
        kind="usr",
        sequence=sequence,
        source=source.label or f"{source.dataset}:{source.record_id}:{source.field}",
        dataset=source.dataset,
        field=source.field,
        record_id=source.record_id,
        circular=bool(cfg.circular),
    )


def _load_path_template(
    *,
    base_dir: Path,
    path_value: str,
    label: str | None,
    sequence_label_prefix: str,
    template_id: str,
    default_id: str | None,
    circular: bool,
    missing_message: str,
    unreadable_message: str,
    empty_message_prefix: str,
    fasta_message_prefix: str,
) -> ResolvedTemplate:
    path = resolve_optional_path(base_dir, path_value)
    if path is None or not path.exists():
        raise ValidationError(missing_message)
    if not path.is_file():
        raise ValidationError(f"{unreadable_message}: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Template path could not be read: {path}") from exc
    seq = _parse_template_text(
        raw,
        path=path,
        empty_message_prefix=empty_message_prefix,
        fasta_message_prefix=fasta_message_prefix,
    )
    return ResolvedTemplate(
        id=template_id or label or default_id or path.stem,
        kind="path",
        sequence=ensure_dna_text(seq, label=f"{sequence_label_prefix} ({path})"),
        source=label or str(path),
        dataset=None,
        field=None,
        record_id=None,
        circular=circular,
    )


def _parse_template_text(
    raw: str,
    *,
    path: Path,
    empty_message_prefix: str,
    fasta_message_prefix: str,
) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValidationError(f"{empty_message_prefix}: {path}")
    if not lines[0].startswith(">"):
        return "".join(lines)

    header_count = sum(1 for line in lines if line.startswith(">"))
    if header_count != 1:
        raise ValidationError(f"{fasta_message_prefix} must contain exactly one record. Found {header_count}: {path}")
    seq_lines = [line for line in lines if not line.startswith(">")]
    if not seq_lines:
        raise ValidationError(f"{fasta_message_prefix} does not contain sequence lines: {path}")
    return "".join(seq_lines)
