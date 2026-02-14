"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/sequence_windows_v1.py

Adapter for sequence-window payloads that map sequence + regulator windows +
motif references into Record features/effects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..core import ContractError, Record, SchemaError, Span, reject_unknown_keys
from ..core.record import Display, Effect, Feature

_RECORD_KEYS = {"id", "sequence", "regulator_windows", "motifs", "display"}
_SEQUENCE_KEYS = {"sense_5to3", "alphabet"}
_WINDOW_KEYS = {"window_id", "tf", "span", "strand", "window_seq_5to3", "score", "meta"}
_WINDOW_SPAN_KEYS = {"start", "end"}
_MOTIF_KEYS = {"tf", "motif_ref", "matrix"}
_MOTIF_REF_KEYS = {"source", "motif_id"}
_DISPLAY_KEYS = {"title", "overlay_text", "tag_labels"}


def _to_mapping(value: Any, *, ctx: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise SchemaError(f"{ctx} is a string but not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} must be a mapping/dict")
    return value


def _to_list(value: Any, *, ctx: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            raise SchemaError(f"{ctx} is a string but not valid JSON") from exc
    if not isinstance(value, list):
        raise SchemaError(f"{ctx} must be a list")
    return value


def _parse_strand(raw: object, *, ctx: str) -> str:
    strand_raw = str(raw).strip().lower()
    if strand_raw in {"fwd", "+"}:
        return "fwd"
    if strand_raw in {"rev", "-"}:
        return "rev"
    raise SchemaError(f"{ctx} must be one of fwd|rev|+|-")


def _coerce_matrix(raw: object, *, ctx: str) -> list[list[float]]:
    if not isinstance(raw, list) or not raw:
        raise SchemaError(f"{ctx} must be a non-empty list")
    rows: list[list[float]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            raise SchemaError(f"{ctx}[{idx}] must have at least 4 values [A,C,G,T]")
        rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return rows


@dataclass(frozen=True)
class SequenceWindowsV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def _parse_sequence(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw.strip().upper()
        data = _to_mapping(raw, ctx="sequence")
        reject_unknown_keys(data, _SEQUENCE_KEYS, "sequence")
        seq = str(data.get("sense_5to3", "")).strip().upper()
        if seq == "":
            raise SchemaError("sequence.sense_5to3 is required")
        seq_alpha = data.get("alphabet")
        if seq_alpha is not None and str(seq_alpha).strip().upper() != str(self.alphabet).upper():
            raise SchemaError(f"sequence.alphabet does not match input.alphabet: {seq_alpha!r} != {self.alphabet!r}")
        return seq

    def _parse_display(self, raw: Any, *, tag_labels: Mapping[str, str]) -> Display:
        if raw is None:
            return Display(overlay_text=None, tag_labels=dict(tag_labels))
        data = _to_mapping(raw, ctx="display")
        reject_unknown_keys(data, _DISPLAY_KEYS, "display")
        merged_labels = dict(tag_labels)
        raw_labels = data.get("tag_labels")
        if raw_labels is not None:
            labels_mapping = _to_mapping(raw_labels, ctx="display.tag_labels")
            merged_labels.update({str(k): str(v) for k, v in labels_mapping.items()})

        overlay = data.get("overlay_text")
        if overlay is None:
            overlay = data.get("title")
        overlay_text = None if overlay is None else str(overlay).strip()
        if overlay_text == "":
            overlay_text = None
        return Display(overlay_text=overlay_text, tag_labels=merged_labels)

    def _parse_motifs(self, raw: Any, *, ctx: str) -> dict[str, dict[str, object]]:
        if raw is None:
            return {}
        motifs_list = _to_list(raw, ctx=ctx)
        out: dict[str, dict[str, object]] = {}
        for idx, item in enumerate(motifs_list):
            motif = _to_mapping(item, ctx=f"{ctx}[{idx}]")
            reject_unknown_keys(motif, _MOTIF_KEYS, f"{ctx}[{idx}]")
            tf_name = str(motif.get("tf", "")).strip()
            if tf_name == "":
                raise SchemaError(f"{ctx}[{idx}].tf is required")
            entry: dict[str, object] = {}

            matrix_raw = motif.get("matrix")
            if matrix_raw is not None:
                entry["matrix"] = _coerce_matrix(matrix_raw, ctx=f"{ctx}[{idx}].matrix")

            motif_ref_raw = motif.get("motif_ref")
            if motif_ref_raw is not None:
                motif_ref = _to_mapping(motif_ref_raw, ctx=f"{ctx}[{idx}].motif_ref")
                reject_unknown_keys(motif_ref, _MOTIF_REF_KEYS, f"{ctx}[{idx}].motif_ref")
                source = str(motif_ref.get("source", "")).strip()
                motif_id = str(motif_ref.get("motif_id", "")).strip()
                if source == "" or motif_id == "":
                    raise SchemaError(f"{ctx}[{idx}].motif_ref requires source and motif_id")
                entry["motif_ref"] = {"source": source, "motif_id": motif_id}

            if not entry:
                raise SchemaError(f"{ctx}[{idx}] requires matrix and/or motif_ref")
            out[tf_name] = entry
            out[tf_name.lower()] = entry
        return out

    def apply(self, row: dict, *, row_index: int) -> Record:
        reject_unknown_keys(row, _RECORD_KEYS, "row")
        sequence_col = str(self.columns.get("sequence"))
        windows_col = str(self.columns.get("regulator_windows"))
        motifs_col = self.columns.get("motifs")
        display_col = self.columns.get("display")
        id_col = self.columns.get("id")

        sequence = self._parse_sequence(row.get(sequence_col))
        windows = _to_list(row.get(windows_col), ctx=windows_col)
        motifs_by_tf = self._parse_motifs(None if motifs_col is None else row.get(str(motifs_col)), ctx=str(motifs_col))

        features: list[Feature] = []
        effects: list[Effect] = []
        tag_labels: dict[str, str] = {}
        for idx, raw_window in enumerate(windows):
            window = _to_mapping(raw_window, ctx=f"{windows_col}[{idx}]")
            reject_unknown_keys(window, _WINDOW_KEYS, f"{windows_col}[{idx}]")

            window_id = str(window.get("window_id", "")).strip()
            if window_id == "":
                raise SchemaError(f"{windows_col}[{idx}].window_id is required")
            tf_name = str(window.get("tf", "")).strip()
            if tf_name == "":
                raise SchemaError(f"{windows_col}[{idx}].tf is required")

            span_data = _to_mapping(window.get("span"), ctx=f"{windows_col}[{idx}].span")
            reject_unknown_keys(span_data, _WINDOW_SPAN_KEYS, f"{windows_col}[{idx}].span")
            if "start" not in span_data or "end" not in span_data:
                raise SchemaError(f"{windows_col}[{idx}].span requires start and end")
            start = int(span_data["start"])
            end = int(span_data["end"])

            strand = _parse_strand(window.get("strand"), ctx=f"{windows_col}[{idx}].strand")
            window_seq = str(window.get("window_seq_5to3", "")).strip().upper()
            if window_seq == "":
                raise SchemaError(f"{windows_col}[{idx}].window_seq_5to3 is required")
            if len(window_seq) != (end - start):
                raise SchemaError(
                    f"{windows_col}[{idx}].window_seq_5to3 length ({len(window_seq)}) "
                    f"must match span length ({end - start})"
                )

            # Reverse-strand windows are rendered below the sequence row where labels
            # are drawn right-to-left; store reversed text so displayed 5'->3' text
            # matches window_seq_5to3.
            label = window_seq if strand == "fwd" else window_seq[::-1]
            tag = f"tf:{tf_name}"
            attrs: dict[str, object] = {
                "tf": tf_name,
                "window_id": window_id,
                "score": window.get("score"),
            }
            raw_meta = window.get("meta")
            if raw_meta is not None:
                attrs["meta"] = dict(_to_mapping(raw_meta, ctx=f"{windows_col}[{idx}].meta"))

            features.append(
                Feature(
                    id=window_id,
                    kind="regulator_window",
                    span=Span(start=start, end=end, strand=strand),
                    label=label,
                    tags=(tag,),
                    attrs=attrs,
                    render={"priority": 10},
                )
            )
            tag_labels.setdefault(tag, tf_name)

            motif = motifs_by_tf.get(tf_name) or motifs_by_tf.get(tf_name.lower())
            if motif is None:
                raise SchemaError(f"missing motif payload for tf='{tf_name}'")
            params: dict[str, object] = {}
            if "matrix" in motif:
                matrix = motif["matrix"]
                if not isinstance(matrix, list):
                    raise SchemaError(f"motif matrix for tf='{tf_name}' must be a list")
                if len(matrix) != (end - start):
                    raise SchemaError(
                        f"motif matrix length ({len(matrix)}) must match window span length ({end - start}) "
                        f"for tf='{tf_name}'"
                    )
                params["matrix"] = matrix
            if "motif_ref" in motif:
                params["motif_ref"] = motif["motif_ref"]
            if not params:
                raise SchemaError(f"motif payload for tf='{tf_name}' has no usable params")

            effects.append(
                Effect(
                    kind="motif_logo",
                    target={"feature_id": window_id},
                    params=params,
                    render={"priority": 20},
                )
            )

        record_id_raw = row.get(str(id_col)) if id_col is not None else None
        record_id = str(record_id_raw).strip() if record_id_raw is not None else ""
        if record_id == "":
            record_id = f"row_{row_index}"

        display = self._parse_display(None if display_col is None else row.get(str(display_col)), tag_labels=tag_labels)
        record = Record(
            id=record_id,
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=display,
            meta={"row_index": row_index, "adapter": "sequence_windows_v1"},
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
