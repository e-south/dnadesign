"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/row_contract.py

Row contracts for study-owned promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from .baserender_validation import validate_baserender_rows
from .contracts import (
    DENSEGEN_RENDER_ANNOTATION_KEYS,
    GENBANK_RENDER_ANNOTATION_KEYS,
    PromoterCandidateBindingsError,
)
from .values import required_sha256, required_text

_IUPAC_DNA = re.compile(r"[ACGTRYSWKMBDHVN]+")
_REGULATOR_OPTIONAL_DESIGN_FAMILIES = frozenset({"background_only", "control"})

BINDING_COLUMNS: tuple[str, ...] = (
    "alias_namespace",
    "alias",
    "display_label",
    "candidate_id",
    "canonical_sequence",
    "sequence_sha256",
    "candidate_table_id",
    "candidate_selection_sha256",
    "sequence_authority_dataset_id",
    "sequence_authority_id",
    "sequence_authority_sha256",
    "source_class",
    "design_family",
    "baserender_adapter_kind",
    "baserender_annotation_column",
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
    "densegen__used_tfbs_detail",
    "densegen__required_regulators",
    "seq_annot__features",
    "seq_annot__source_file",
    "usr_label__primary",
    "derived__product_kind",
    "binding_status",
    "binding_method",
)
_CANDIDATE_INVARIANT_COLUMNS = tuple(
    column
    for column in BINDING_COLUMNS
    if column
    not in {
        "alias_namespace",
        "alias",
        "display_label",
        "sequence_authority_dataset_id",
        "sequence_authority_id",
        "sequence_authority_sha256",
        "binding_status",
        "binding_method",
    }
)


def validate_binding_rows(rows: pd.DataFrame) -> None:
    if tuple(rows.columns) != BINDING_COLUMNS or rows.empty:
        raise PromoterCandidateBindingsError("Binding rows must be non-empty and match the v1 column contract.")
    duplicates = rows.duplicated(subset=["alias_namespace", "alias"], keep=False)
    if duplicates.any():
        raise PromoterCandidateBindingsError("Binding aliases must be unique within each namespace.")
    for _, row in rows.iterrows():
        _validate_binding_row(row)
    _validate_candidate_invariants(rows)
    validate_baserender_rows(rows)


def _validate_binding_row(row: pd.Series) -> None:
    namespace = required_text(row["alias_namespace"], field="alias namespace")
    alias = required_text(row["alias"], field="alias")
    identity = f"{namespace}:{alias}"
    required_text(row["display_label"], field="display label", row_id=identity)
    required_text(row["candidate_id"], field="candidate ID", row_id=identity)
    sequence = required_text(row["canonical_sequence"], field="canonical sequence", row_id=identity)
    if sequence != sequence.upper() or _IUPAC_DNA.fullmatch(sequence) is None:
        raise PromoterCandidateBindingsError(f"Binding {identity!r} canonical sequence must be uppercase IUPAC DNA.")
    digest = required_sha256(row["sequence_sha256"], field="sequence SHA-256")
    if digest != hashlib.sha256(sequence.encode()).hexdigest():
        raise PromoterCandidateBindingsError(f"Binding {identity!r} canonical sequence digest mismatch.")
    for field in (
        "candidate_table_id",
        "sequence_authority_dataset_id",
        "sequence_authority_id",
        "source_class",
        "design_family",
    ):
        required_text(row[field], field=field, row_id=identity)
    required_sha256(row["candidate_selection_sha256"], field="candidate selection SHA-256")
    required_sha256(row["sequence_authority_sha256"], field="sequence authority SHA-256")
    if row["binding_status"] != "resolved" or row["binding_method"] != "exact_alias":
        raise PromoterCandidateBindingsError(f"Binding {identity!r} must use exact resolved alias semantics.")
    adapter = row["baserender_adapter_kind"]
    if adapter == "densegen_tfbs":
        _validate_densegen(row, sequence=sequence, identity=identity)
    elif adapter == "usr_genbank_annotations_v1":
        _validate_genbank(row, sequence=sequence, identity=identity)
    else:
        raise PromoterCandidateBindingsError(f"Binding {identity!r} has unsupported adapter {adapter!r}.")


def _validate_densegen(row: pd.Series, *, sequence: str, identity: str) -> None:
    if row["baserender_annotation_column"] != "densegen__used_tfbs_detail":
        raise PromoterCandidateBindingsError(f"Binding {identity!r} has the wrong DenseGen annotation column.")
    for field in ("densegen__plan", "densegen__run_id", "densegen__sampling_library_hash"):
        required_text(row[field], field=field, row_id=identity)
    regulators = [
        required_text(value, field="required regulator")
        for value in _sequence(row["densegen__required_regulators"], field="required regulators")
    ]
    design_family = str(row["design_family"]).strip()
    if not regulators and design_family not in _REGULATOR_OPTIONAL_DESIGN_FAMILIES:
        raise PromoterCandidateBindingsError(
            f"Binding {identity!r} design family {design_family!r} requires at least one DenseGen regulator."
        )
    if len(regulators) != len(set(regulators)):
        raise PromoterCandidateBindingsError(f"Binding {identity!r} contains duplicate DenseGen regulators.")
    annotations = _sequence(row["densegen__used_tfbs_detail"], field="DenseGen annotations")
    if not annotations:
        raise PromoterCandidateBindingsError(f"Binding {identity!r} requires DenseGen annotations.")
    for index, item in enumerate(annotations):
        record = _annotation(item, allowed=DENSEGEN_RENDER_ANNOTATION_KEYS, identity=identity, index=index)
        kind = required_text(record.get("part_kind"), field=f"DenseGen annotation {index}.part_kind")
        literal = required_text(record.get("sequence"), field=f"DenseGen annotation {index}.sequence")
        if kind == "tfbs":
            start, length, end = (_integer(record.get(field), field=field) for field in ("offset", "length", "end"))
            if start < 0 or length < 1 or end != start + length or end > len(sequence) or len(literal) != length:
                raise PromoterCandidateBindingsError(f"Binding {identity!r} has an invalid TFBS span.")
            if record.get("orientation") not in {"fwd", "rev"}:
                raise PromoterCandidateBindingsError(f"Binding {identity!r} has an invalid TFBS orientation.")
            required_text(record.get("regulator"), field="TFBS regulator")
        elif kind == "fixed_element":
            if record.get("role") not in {"upstream", "downstream"}:
                raise PromoterCandidateBindingsError(f"Binding {identity!r} has an invalid fixed-element role.")
            required_text(record.get("constraint_name"), field="fixed-element constraint name")
        else:
            raise PromoterCandidateBindingsError(f"Binding {identity!r} has an unsupported DenseGen part kind.")


def _validate_genbank(row: pd.Series, *, sequence: str, identity: str) -> None:
    if row["baserender_annotation_column"] != "seq_annot__features":
        raise PromoterCandidateBindingsError(f"Binding {identity!r} has the wrong GenBank annotation column.")
    densegen_fields = (
        "densegen__plan",
        "densegen__run_id",
        "densegen__sampling_library_hash",
        "densegen__used_tfbs_detail",
        "densegen__required_regulators",
    )
    if any(not _missing(row[field]) for field in densegen_fields):
        raise PromoterCandidateBindingsError(f"Binding {identity!r} mixes GenBank and DenseGen metadata.")
    annotations = _sequence(row["seq_annot__features"], field="GenBank annotations")
    if not annotations:
        raise PromoterCandidateBindingsError(f"Binding {identity!r} requires GenBank annotations.")
    for index, item in enumerate(annotations):
        record = _annotation(item, allowed=GENBANK_RENDER_ANNOTATION_KEYS, identity=identity, index=index)
        for field in ("feature_id", "feature_type", "label"):
            required_text(record.get(field), field=f"GenBank annotation {index}.{field}")
        start, end, strand = (_integer(record.get(field), field=field) for field in ("start_0", "end_0", "strand"))
        if start < 0 or end <= start or end > len(sequence) or strand not in {-1, 0, 1}:
            raise PromoterCandidateBindingsError(f"Binding {identity!r} has an invalid GenBank span.")
    _relative_path(row["seq_annot__source_file"], field="GenBank source artifact")


def _validate_candidate_invariants(rows: pd.DataFrame) -> None:
    for candidate_id, group in rows.groupby("candidate_id", sort=False):
        for column in _CANDIDATE_INVARIANT_COLUMNS:
            values = {_stable(value) for value in group[column].tolist()}
            if len(values) != 1:
                raise PromoterCandidateBindingsError(
                    f"Candidate {candidate_id!r} has alias-dependent {column!r} metadata."
                )


def _annotation(value: object, *, allowed: tuple[str, ...], identity: str, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromoterCandidateBindingsError(f"Binding {identity!r} annotation {index} must be a mapping.")
    extras = sorted(set(value) - set(allowed))
    if extras:
        raise PromoterCandidateBindingsError(f"Binding {identity!r} has non-contract annotation fields: {extras}")
    return value


def _sequence(value: object, *, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PromoterCandidateBindingsError(f"{field} must be a sequence.")
    return list(value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise PromoterCandidateBindingsError(f"{field} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PromoterCandidateBindingsError(f"{field} must be an integer.") from exc
    if number != value:
        raise PromoterCandidateBindingsError(f"{field} must be an integer.")
    return number


def _relative_path(value: object, *, field: str) -> str:
    text = required_text(value, field=field)
    path = PurePosixPath(text)
    first = path.parts[0] if path.parts else ""
    if "\\" in text or text.startswith("~") or path.is_absolute() or ".." in path.parts or ":" in first:
        raise PromoterCandidateBindingsError(f"{field} must be a confined relative POSIX path.")
    return str(path)


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _stable(value: object) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _missing(value):
        value = None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
