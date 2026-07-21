"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/genbank.py

GenBank projection for stress-study synthesis handoff manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from .contracts import require_nonnegative_integer

GENBANK_FEATURE_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "campaign_slug",
    "id",
    "synthesis_name",
    "feature_type",
    "feature_id",
    "label",
    "start_0",
    "end_0",
    "strand",
    "sequence",
    "source",
    "densegen_coordinate_key",
    "densegen_offset",
    "densegen_offset_raw",
    "densegen_orientation",
    "densegen_expected_sequence",
    "densegen_part_kind",
    "regulator",
    "role",
    "constraint_name",
    "variant_id",
    "spacer_length",
    "motif_id",
    "tfbs_id",
    "score_relative_to_theoretical_max",
    "tier",
    "genbank_location",
)

_REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "campaign_slug",
    "selection_view_ids",
    "selection_memberships",
    "id",
    "synthesis_name",
    "core_sequence",
    "core_start",
    "core_end",
    "final_sequence",
)
_DENSEGEN_DETAIL_COLUMN = "densegen__used_tfbs_detail"
_REGULATOR_DISPLAY = {"baeR": "BaeR", "cpxR": "CpxR", "lexA": "LexA"}
_SAFE_LOCUS = re.compile(r"[^A-Za-z0-9_.-]+")
_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _require_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError("synthesis manifest missing required GenBank columns: " + ", ".join(missing))


def _genbank_location(start: int, end: int, *, strand: int = 1) -> str:
    location = f"{start + 1}..{end}"
    if int(strand) == -1:
        return f"complement({location})"
    return location


def _feature_row(
    manifest_row: pd.Series,
    *,
    feature_type: str,
    feature_id: str,
    label: str,
    start_0: int,
    end_0: int,
    strand: int = 1,
    source: str,
    densegen_coordinate_key: str = "",
    densegen_offset: Any = "",
    densegen_offset_raw: Any = "",
    densegen_orientation: str = "",
    densegen_expected_sequence: str = "",
    densegen_part_kind: str = "",
    regulator: str = "",
    role: str = "",
    constraint_name: str = "",
    variant_id: str = "",
    spacer_length: Any = "",
    motif_id: str = "",
    tfbs_id: str = "",
    score_relative_to_theoretical_max: Any = "",
    tier: Any = "",
) -> dict[str, Any]:
    final_sequence = str(manifest_row["final_sequence"])
    start = int(start_0)
    end = int(end_0)
    if start < 0 or end <= start or end > len(final_sequence):
        raise ValueError(
            f"GenBank feature bounds invalid for {manifest_row['id']}: "
            f"{feature_id} [{start}, {end}) length={len(final_sequence)}"
        )
    feature_sequence = final_sequence[start:end].upper()
    display_strand = int(strand)
    return {
        "batch_id": str(manifest_row["batch_id"]),
        "campaign_slug": str(manifest_row["campaign_slug"]),
        "id": str(manifest_row["id"]),
        "synthesis_name": str(manifest_row["synthesis_name"]),
        "feature_type": str(feature_type),
        "feature_id": str(feature_id),
        "label": str(label),
        "start_0": start,
        "end_0": end,
        "strand": display_strand,
        "sequence": feature_sequence,
        "source": str(source),
        "densegen_coordinate_key": str(densegen_coordinate_key),
        "densegen_offset": "" if pd.isna(densegen_offset) else str(densegen_offset),
        "densegen_offset_raw": "" if pd.isna(densegen_offset_raw) else str(densegen_offset_raw),
        "densegen_orientation": str(densegen_orientation or ""),
        "densegen_expected_sequence": str(densegen_expected_sequence or "").upper(),
        "densegen_part_kind": str(densegen_part_kind),
        "regulator": str(regulator),
        "role": str(role),
        "constraint_name": str(constraint_name),
        "variant_id": "" if pd.isna(variant_id) else str(variant_id),
        "spacer_length": "" if pd.isna(spacer_length) else str(spacer_length),
        "motif_id": str(motif_id or ""),
        "tfbs_id": str(tfbs_id or ""),
        "score_relative_to_theoretical_max": ""
        if pd.isna(score_relative_to_theoretical_max)
        else str(score_relative_to_theoretical_max),
        "tier": "" if pd.isna(tier) else str(tier),
        "genbank_location": _genbank_location(start, end, strand=display_strand),
    }


def _candidate_detail_map(candidate_records_path: str | Path | None, ids: set[str]) -> dict[str, dict[str, Any]]:
    if candidate_records_path is None:
        return {}
    path = Path(candidate_records_path)
    if not path.exists():
        raise ValueError(f"candidate records parquet not found for GenBank annotations: {path}")
    try:
        rows = pd.read_parquet(path, columns=["id", "sequence", _DENSEGEN_DETAIL_COLUMN])
    except Exception as exc:
        raise ValueError(
            "candidate records parquet must expose id, sequence, and "
            f"{_DENSEGEN_DETAIL_COLUMN} for GenBank annotations: {path}"
        ) from exc
    rows["id"] = rows["id"].astype(str)
    duplicated = rows.loc[rows["id"].duplicated(), "id"].astype(str).tolist()
    if duplicated:
        raise ValueError(f"candidate records contain duplicate ids for GenBank annotation join: {duplicated[0]}")
    found = set(rows["id"].tolist())
    missing = sorted(ids.difference(found))
    if missing:
        missing_preview = ", ".join(missing[:5])
        raise ValueError(f"candidate records missing selected ids for GenBank annotation join: {missing_preview}")
    indexed = rows.set_index("id")
    return {
        str(candidate_id): {
            "sequence": str(row["sequence"]),
            "densegen_detail": row[_DENSEGEN_DETAIL_COLUMN],
        }
        for candidate_id, row in indexed.iterrows()
        if str(candidate_id) in ids
    }


def _detail_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        value = json.loads(text)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"DenseGen annotation detail must be a list, got {type(value).__name__}")
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"DenseGen annotation detail item {index} must be a mapping, got {type(item).__name__}")
        items.append(item)
    return items


def _regulator_base(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    for regulator in ("baeR", "cpxR", "lexA"):
        if raw.startswith(regulator):
            return regulator
    if raw == "background":
        return "background"
    return raw.split("_", 1)[0]


def _display_regulator(regulator: str) -> str:
    return _REGULATOR_DISPLAY.get(regulator, regulator[:1].upper() + regulator[1:])


def _strand_from_orientation(value: Any) -> int:
    if str(value or "").strip().lower() in {"rev", "reverse", "reverse_complement", "-1"}:
        return -1
    return 1


def _tfbs_strand_from_orientation(value: Any, *, candidate_id: str) -> int:
    orientation = str(value or "").strip().lower()
    if orientation in {"fwd", "forward", "1"}:
        return 1
    if orientation in {"rev", "reverse", "reverse_complement", "-1"}:
        return -1
    raise ValueError(f"DenseGen TFBS annotation for {candidate_id} requires explicit fwd/rev orientation")


def _reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(_DNA_COMPLEMENT)[::-1]


def _required_densegen_sequence(item: dict[str, Any], *, candidate_id: str) -> str:
    expected_sequence = str(item.get("sequence") or "").upper()
    if not expected_sequence:
        raise ValueError(f"DenseGen annotation for {candidate_id} requires non-empty sequence")
    return expected_sequence


def _required_densegen_coordinate(item: dict[str, Any], *, key: str, candidate_id: str) -> int:
    value = item.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"DenseGen annotation for {candidate_id} requires coordinate key {key}")
    return require_nonnegative_integer(
        value,
        field=f"DenseGen annotation {key} for {candidate_id}",
    )


def _densegen_coordinate_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "densegen_offset": item.get("offset", ""),
        "densegen_offset_raw": item.get("offset_raw", ""),
        "densegen_orientation": str(item.get("orientation") or ""),
        "densegen_expected_sequence": str(item.get("sequence") or "").upper(),
    }


def _densegen_span(
    item: dict[str, Any],
    *,
    core_sequence: str,
    candidate_id: str,
    coordinate_key: str,
    strand: int,
    end_policy: str,
) -> tuple[int, int]:
    expected_sequence = _required_densegen_sequence(item, candidate_id=candidate_id)
    start = _required_densegen_coordinate(item, key=coordinate_key, candidate_id=candidate_id)
    if end_policy == "densegen_end":
        if item.get("end") is None or pd.isna(item.get("end")):
            raise ValueError(
                f"DenseGen annotation for {candidate_id} requires end with coordinate key {coordinate_key}"
            )
        end = require_nonnegative_integer(
            item["end"],
            field=f"DenseGen annotation end for {candidate_id}",
        )
        expected_end = start + len(expected_sequence)
        if end != expected_end:
            raise ValueError(
                f"DenseGen annotation end mismatch for {candidate_id}: "
                f"coordinate key {coordinate_key} expected end {expected_end}, observed {end}"
            )
    elif end_policy == "sequence_length":
        end = start + len(expected_sequence)
    else:
        raise ValueError(f"unsupported DenseGen coordinate end policy: {end_policy}")
    if start < 0 or end <= start or end > len(core_sequence):
        raise ValueError(
            f"DenseGen annotation bounds invalid for {candidate_id}: "
            f"coordinate key {coordinate_key} [{start}, {end}) length={len(core_sequence)}"
        )
    expected_observed = _reverse_complement(expected_sequence) if int(strand) == -1 else expected_sequence
    observed = core_sequence[start:end].upper()
    if observed == expected_observed:
        return start, end
    raise ValueError(
        "DenseGen annotation sequence mismatch for "
        f"{candidate_id}: required coordinate key {coordinate_key} expected {expected_observed!r} "
        f"at [{start}, {end}), observed {observed!r}"
    )


def _densegen_feature_rows(manifest_row: pd.Series, detail: Any) -> list[dict[str, Any]]:
    core_sequence = str(manifest_row["core_sequence"]).upper()
    core_start = int(manifest_row["core_start"])
    candidate_id = str(manifest_row["id"])
    feature_prefix = str(manifest_row["synthesis_name"])
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(_detail_items(detail)):
        part_kind = str(item.get("part_kind") or "").strip()
        if part_kind == "tfbs":
            regulator = _regulator_base(item.get("regulator"))
            if not regulator:
                raise ValueError(f"DenseGen TFBS annotation for {candidate_id} requires a regulator")
            if regulator == "background":
                continue
            strand = _tfbs_strand_from_orientation(item.get("orientation"), candidate_id=candidate_id)
            start, end = _densegen_span(
                item,
                core_sequence=core_sequence,
                candidate_id=candidate_id,
                coordinate_key="offset",
                strand=strand,
                end_policy="densegen_end",
            )
            rows.append(
                _feature_row(
                    manifest_row,
                    feature_type="misc_feature",
                    feature_id=f"{feature_prefix}:tfbs:{idx}",
                    label=f"{_display_regulator(regulator)} TFBS",
                    start_0=core_start + start,
                    end_0=core_start + end,
                    strand=strand,
                    source="densegen_tfbs",
                    densegen_coordinate_key="offset",
                    **_densegen_coordinate_metadata(item),
                    densegen_part_kind=part_kind,
                    regulator=regulator,
                    motif_id=str(item.get("motif_id") or ""),
                    tfbs_id=str(item.get("tfbs_id") or ""),
                    score_relative_to_theoretical_max=item.get("score_relative_to_theoretical_max", ""),
                    tier=item.get("tier", ""),
                )
            )
            continue
        if part_kind == "fixed_element":
            constraint_name = str(item.get("constraint_name") or "").strip()
            if constraint_name != "sigma70_core":
                raise ValueError(
                    f"unsupported DenseGen fixed_element constraint for {candidate_id}: {constraint_name!r}"
                )
            role = str(item.get("role") or "").strip()
            if role not in {"upstream", "downstream"}:
                raise ValueError(f"unsupported DenseGen sigma70_core role for {candidate_id}: {role!r}")
            variant_id = str(item.get("variant_id") or "").strip()
            if role == "upstream":
                label = f"-35 ({variant_id})" if variant_id else "-35"
            elif role == "downstream":
                label = f"-10 ({variant_id})" if variant_id else "-10"
            else:
                label = "sigma70 core"
            start, end = _densegen_span(
                item,
                core_sequence=core_sequence,
                candidate_id=candidate_id,
                coordinate_key="offset_raw",
                strand=1,
                end_policy="sequence_length",
            )
            rows.append(
                _feature_row(
                    manifest_row,
                    feature_type="misc_feature",
                    feature_id=f"{feature_prefix}:sigma70:{role or idx}",
                    label=label,
                    start_0=core_start + start,
                    end_0=core_start + end,
                    source="densegen_fixed_element",
                    densegen_coordinate_key="offset_raw",
                    **_densegen_coordinate_metadata(item),
                    densegen_part_kind=part_kind,
                    role=role,
                    constraint_name="sigma70_core",
                    variant_id=variant_id,
                    spacer_length=item.get("spacer_length", ""),
                )
            )
            continue
        raise ValueError(f"unsupported DenseGen annotation part_kind for {candidate_id}: {part_kind!r}")
    return rows


def build_genbank_feature_table(
    manifest: pd.DataFrame,
    *,
    candidate_records_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build feature rows for annotated synthesis GenBank exports."""

    _require_manifest_columns(manifest)
    details = _candidate_detail_map(candidate_records_path, set(manifest["id"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for _, manifest_row in manifest.iterrows():
        core_start = int(manifest_row["core_start"])
        core_end = int(manifest_row["core_end"])
        final_length = len(str(manifest_row["final_sequence"]))
        feature_prefix = str(manifest_row["synthesis_name"])
        rows.extend(
            [
                _feature_row(
                    manifest_row,
                    feature_type="source",
                    feature_id=f"{feature_prefix}:source",
                    label=str(manifest_row["synthesis_name"]),
                    start_0=0,
                    end_0=final_length,
                    source="synthesis_handoff",
                ),
                _feature_row(
                    manifest_row,
                    feature_type="misc_feature",
                    feature_id=f"{feature_prefix}:left_flank",
                    label="5' cloning flank",
                    start_0=0,
                    end_0=core_start,
                    source="cloning_strategy",
                ),
                _feature_row(
                    manifest_row,
                    feature_type="promoter",
                    feature_id=f"{feature_prefix}:promoter_core",
                    label="60 nt promoter core",
                    start_0=core_start,
                    end_0=core_end,
                    source="opal_candidate",
                ),
                _feature_row(
                    manifest_row,
                    feature_type="misc_feature",
                    feature_id=f"{feature_prefix}:right_flank",
                    label="3' cloning flank",
                    start_0=core_end,
                    end_0=final_length,
                    source="cloning_strategy",
                ),
            ]
        )
        detail = details.get(str(manifest_row["id"]))
        if detail is not None:
            if str(detail["sequence"]).upper() != str(manifest_row["core_sequence"]).upper():
                raise ValueError(
                    f"candidate record sequence mismatch for GenBank annotation join: {manifest_row['id']}"
                )
            rows.extend(_densegen_feature_rows(manifest_row, detail["densegen_detail"]))
    return pd.DataFrame(rows, columns=list(GENBANK_FEATURE_COLUMNS))


def _safe_locus(value: Any) -> str:
    text = _SAFE_LOCUS.sub("_", str(value or "").strip())
    if not text:
        text = "synthesis_insert"
    return text[:16]


def _safe_filename_token(value: Any) -> str:
    text = _SAFE_LOCUS.sub("_", str(value or "").strip())
    if not text:
        raise ValueError("GenBank filename token must be non-empty")
    return text


def genbank_record_filename(manifest_row: pd.Series) -> str:
    """Return the detached-safe filename for one synthesis insert GenBank file."""

    return (
        f"{_safe_filename_token(manifest_row['batch_id'])}__"
        f"{_safe_filename_token(manifest_row['campaign_slug'])}__"
        f"{_safe_filename_token(manifest_row['synthesis_name'])}__annotated_insert.gb"
    )


def _selection_qualifiers(manifest_row: pd.Series) -> dict[str, list[str]]:
    try:
        view_ids = json.loads(str(manifest_row["selection_view_ids"]))
        memberships = json.loads(str(manifest_row["selection_memberships"]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest selection membership JSON is invalid for {manifest_row['id']}") from exc
    if not isinstance(view_ids, list) or not isinstance(memberships, list):
        raise ValueError(f"manifest selection membership fields must be JSON lists for {manifest_row['id']}")
    membership_values = []
    for row in memberships:
        if not isinstance(row, dict):
            raise ValueError(f"manifest selection membership rows must be mappings for {manifest_row['id']}")
        parts = [f"view={row.get('selection_view_id')}", f"rank={row.get('rank')}"]
        if row.get("score") is not None:
            parts.append(f"score={row['score']}")
        if row.get("score_ref") is not None:
            parts.append(f"score_ref={row['score_ref']}")
        membership_values.append("|".join(parts))
    return {
        "selection_views": [",".join(str(view_id) for view_id in view_ids)],
        "selection_membership": membership_values,
    }


def _feature_qualifiers(manifest_row: pd.Series, feature_row: pd.Series) -> dict[str, list[str]]:
    qualifiers: dict[str, list[str]] = {
        "label": [str(feature_row["label"])],
        "campaign_slug": [str(manifest_row["campaign_slug"])],
        **_selection_qualifiers(manifest_row),
        "batch_id": [str(manifest_row["batch_id"])],
        "handoff_id": [str(manifest_row["batch_id"])],
        "synthesis_name": [str(manifest_row["synthesis_name"])],
        "canonical_id": [str(manifest_row["id"])],
        "strategy_id": [str(manifest_row.get("strategy_id", ""))],
        "core_sha256": [str(manifest_row.get("core_sha256", ""))],
        "final_sha256": [str(manifest_row.get("final_sha256", ""))],
        "dnadesign_feature_id": [str(feature_row["feature_id"])],
        "dnadesign_source": [str(feature_row["source"])],
    }
    for key in ("selection_epoch", "assay_batch_index", "model_as_of_round", "run_id"):
        value = _optional_qualifier_text(manifest_row.get(key))
        if value:
            qualifiers[key] = [value]
    for key in (
        "densegen_coordinate_key",
        "densegen_offset",
        "densegen_offset_raw",
        "densegen_orientation",
        "densegen_expected_sequence",
        "densegen_part_kind",
        "regulator",
        "role",
        "constraint_name",
        "variant_id",
        "spacer_length",
        "motif_id",
        "tfbs_id",
        "score_relative_to_theoretical_max",
        "tier",
    ):
        value = _optional_qualifier_text(feature_row.get(key))
        if value:
            qualifier_key = {
                "densegen_coordinate_key": "dg_coord_key",
                "densegen_offset": "dg_offset",
                "densegen_offset_raw": "dg_offset_raw",
                "densegen_orientation": "dg_orientation",
                "densegen_expected_sequence": "dg_expected_seq",
                "densegen_part_kind": "dg_part_kind",
                "constraint_name": "dg_constraint",
                "score_relative_to_theoretical_max": "dg_rel_score",
                "spacer_length": "spacer_len",
            }.get(key, key)
            qualifiers[qualifier_key] = [value]
    constraint = str(feature_row.get("constraint_name", "") or "").strip()
    role = str(feature_row.get("role", "") or "").strip()
    variant_id = str(feature_row.get("variant_id", "") or "").strip()
    if constraint == "sigma70_core" and variant_id:
        feature_sequence = str(feature_row.get("sequence", "") or "").strip().upper()
        if role == "upstream":
            qualifiers["sigma35_variant"] = [variant_id]
            if feature_sequence:
                qualifiers["sigma35_sequence"] = [feature_sequence]
        elif role == "downstream":
            qualifiers["sigma10_variant"] = [variant_id]
            if feature_sequence:
                qualifiers["sigma10_sequence"] = [feature_sequence]
    return qualifiers


def _record_from_manifest_row(manifest_row: pd.Series, feature_rows: pd.DataFrame) -> SeqRecord:
    sequence = str(manifest_row["final_sequence"])
    locus = _safe_locus(manifest_row["synthesis_name"])
    record = SeqRecord(
        Seq(sequence),
        id=locus,
        name=locus,
        description=f"{manifest_row['synthesis_name']} {manifest_row['campaign_slug']} {manifest_row['batch_id']}",
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "linear"
    record.annotations["data_file_division"] = "SYN"
    record.annotations["date"] = "01-JAN-2000"
    features: list[SeqFeature] = []
    for _, feature_row in feature_rows.iterrows():
        location = SimpleLocation(
            int(feature_row["start_0"]),
            int(feature_row["end_0"]),
            strand=int(feature_row["strand"]),
        )
        features.append(
            SeqFeature(
                location=location,
                type=str(feature_row["feature_type"]),
                qualifiers=_feature_qualifiers(manifest_row, feature_row),
            )
        )
    record.features = features
    return record


def _optional_qualifier_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def render_genbank_record_set(
    manifest: pd.DataFrame,
    feature_table: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write one GenBank file per synthesis insert under a campaign-local directory."""

    _require_manifest_columns(manifest)
    genbank_dir = Path(output_dir)
    genbank_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in genbank_dir.glob("*.gb"):
        stale_path.unlink()
    rows: list[dict[str, Any]] = []
    for _, manifest_row in manifest.iterrows():
        record_features = feature_table.loc[feature_table["id"].astype(str) == str(manifest_row["id"])].reset_index(
            drop=True
        )
        if record_features.empty:
            raise ValueError(f"GenBank feature table has no rows for {manifest_row['id']}")
        record = _record_from_manifest_row(manifest_row, record_features)
        record_path = genbank_dir / genbank_record_filename(manifest_row)
        with record_path.open("w", encoding="utf-8") as handle:
            SeqIO.write([record], handle, "genbank")
        rows.append(
            {
                "campaign_slug": str(manifest_row["campaign_slug"]),
                "batch_id": str(manifest_row["batch_id"]),
                "id": str(manifest_row["id"]),
                "synthesis_name": str(manifest_row["synthesis_name"]),
                "genbank_file_path": str(record_path),
            }
        )
    return pd.DataFrame(rows)


def read_genbank_records(path: str | Path) -> list[SeqRecord]:
    """Read GenBank records from a synthesis handoff export."""

    genbank_path = Path(path)
    if not genbank_path.exists():
        raise ValueError(f"GenBank export not found: {genbank_path}")
    return list(SeqIO.parse(genbank_path, "genbank"))


def _normalized_qualifier_values(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text = "".join(str(value).split())
        integral_float = re.fullmatch(r"(-?\d+)\.0+", text)
        normalized.append(integral_float.group(1) if integral_float else text)
    return normalized


def _validate_qualifier_values(
    feature: SeqFeature,
    *,
    qualifier: str,
    expected_values: list[str],
    identity: str,
) -> None:
    observed_values = feature.qualifiers.get(qualifier) or []
    if _normalized_qualifier_values(observed_values) != _normalized_qualifier_values(expected_values):
        raise ValueError(
            f"GenBank feature {qualifier} mismatch for {identity}: "
            f"expected {expected_values!r}, observed {observed_values!r}"
        )


def _validate_feature_table_parity(
    record: SeqRecord,
    *,
    expected: pd.Series,
    expected_features: pd.DataFrame,
) -> None:
    identity = str(expected["synthesis_name"])
    observed_by_id: dict[str, SeqFeature] = {}
    for feature in record.features:
        feature_ids = feature.qualifiers.get("dnadesign_feature_id") or []
        if len(feature_ids) != 1:
            raise ValueError(f"GenBank feature missing unique dnadesign_feature_id for {identity}")
        feature_id = "".join(str(feature_ids[0]).split())
        if feature_id in observed_by_id:
            raise ValueError(f"GenBank record contains duplicate feature ID for {identity}: {feature_id}")
        observed_by_id[feature_id] = feature
    expected_ids = {"".join(str(value).split()) for value in expected_features["feature_id"].tolist()}
    if len(expected_ids) != len(expected_features):
        raise ValueError(f"GenBank feature table contains duplicate normalized feature IDs for {identity}")
    if set(observed_by_id) != expected_ids:
        missing = sorted(expected_ids.difference(observed_by_id))
        extra = sorted(set(observed_by_id).difference(expected_ids))
        raise ValueError(f"GenBank feature ID parity mismatch for {identity}: missing={missing[:5]}, extra={extra[:5]}")
    for _, feature_row in expected_features.iterrows():
        feature_id = str(feature_row["feature_id"])
        feature = observed_by_id["".join(feature_id.split())]
        if feature.type != str(feature_row["feature_type"]):
            raise ValueError(f"GenBank feature type mismatch for {identity}/{feature_id}")
        observed_location = (
            int(feature.location.start),
            int(feature.location.end),
            int(feature.location.strand or 0),
        )
        expected_location = (
            int(feature_row["start_0"]),
            int(feature_row["end_0"]),
            int(feature_row["strand"]),
        )
        if observed_location != expected_location:
            raise ValueError(
                f"GenBank feature location mismatch for {identity}/{feature_id}: "
                f"expected {expected_location}, observed {observed_location}"
            )
        expected_qualifiers = _feature_qualifiers(expected, feature_row)
        if set(feature.qualifiers) != set(expected_qualifiers):
            raise ValueError(f"GenBank feature qualifier-key parity mismatch for {identity}/{feature_id}")
        for qualifier, expected_values in expected_qualifiers.items():
            _validate_qualifier_values(
                feature,
                qualifier=qualifier,
                expected_values=expected_values,
                identity=f"{identity}/{feature_id}",
            )


def _validate_record_against_manifest_row(
    record: SeqRecord,
    expected: pd.Series,
    *,
    expected_features: pd.DataFrame | None = None,
) -> None:
    expected_locus = _safe_locus(expected["synthesis_name"])
    if record.id != expected_locus or record.name != expected_locus:
        raise ValueError(
            f"GenBank record identity mismatch for {expected['synthesis_name']}: "
            f"expected {expected_locus!r}, observed id={record.id!r}, name={record.name!r}"
        )
    source = next((feature for feature in record.features if feature.type == "source"), None)
    if source is None:
        raise ValueError(f"GenBank record missing source feature: {record.id}")
    aliases = source.qualifiers.get("synthesis_name") or []
    if aliases != [str(expected["synthesis_name"])]:
        raise ValueError(
            "GenBank record source feature synthesis_name mismatch: "
            f"expected {expected['synthesis_name']!r}, observed {aliases!r}"
        )
    if expected_features is None:
        expected_source = pd.Series(
            {
                "feature_id": f"{expected['synthesis_name']}:source",
                "source": "synthesis_handoff",
                "label": str(expected["synthesis_name"]),
            }
        )
    else:
        source_rows = expected_features.loc[expected_features["feature_type"].astype(str) == "source"]
        if len(source_rows) != 1:
            raise ValueError(f"GenBank feature table must contain one source row for {expected['synthesis_name']}")
        expected_source = source_rows.iloc[0]
    for qualifier, expected_values in _feature_qualifiers(expected, expected_source).items():
        _validate_qualifier_values(
            source,
            qualifier=qualifier,
            expected_values=expected_values,
            identity=str(expected["synthesis_name"]),
        )
    if str(record.seq).upper() != str(expected["final_sequence"]).upper():
        raise ValueError(f"GenBank sequence mismatch for {expected['synthesis_name']}")
    labels = {str(value) for feature in record.features for value in feature.qualifiers.get("label", [])}
    required = {"5' cloning flank", "60 nt promoter core", "3' cloning flank"}
    missing = sorted(required.difference(labels))
    if missing:
        raise ValueError(
            f"GenBank record {expected['synthesis_name']} missing required feature labels: {', '.join(missing)}"
        )
    if expected_features is not None:
        _validate_feature_table_parity(record, expected=expected, expected_features=expected_features)


def validate_genbank_record_set(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_table: pd.DataFrame | str | Path | None = None,
) -> dict[str, Any]:
    """Validate the one-file-per-synthesis-insert GenBank directory."""

    _require_manifest_columns(manifest)
    genbank_dir = Path(output_dir)
    if not genbank_dir.is_dir():
        raise ValueError(f"GenBank record-set directory not found: {genbank_dir}")
    expected_features: pd.DataFrame | None = None
    if feature_table is not None:
        if isinstance(feature_table, (str, Path)):
            feature_table_path = Path(feature_table)
            if not feature_table_path.is_file():
                raise ValueError(f"GenBank feature table not found: {feature_table_path}")
            try:
                expected_features = pd.read_csv(
                    feature_table_path,
                    dtype=str,
                    keep_default_na=False,
                )
            except OSError as exc:
                raise ValueError(f"Could not read GenBank feature table: {feature_table_path}") from exc
        else:
            expected_features = feature_table.copy().fillna("")
        _validate_feature_table_identity(manifest, expected_features)
    expected_filenames = [genbank_record_filename(row) for _, row in manifest.iterrows()]
    observed_filenames = sorted(path.name for path in genbank_dir.glob("*.gb"))
    if observed_filenames != sorted(expected_filenames):
        missing = sorted(set(expected_filenames).difference(observed_filenames))
        extra = sorted(set(observed_filenames).difference(expected_filenames))
        parts: list[str] = []
        if missing:
            parts.append("missing " + ", ".join(missing[:5]))
        if extra:
            parts.append("unexpected " + ", ".join(extra[:5]))
        raise ValueError("GenBank record-set file mismatch: " + "; ".join(parts))
    for _, expected in manifest.iterrows():
        path = genbank_dir / genbank_record_filename(expected)
        records = read_genbank_records(path)
        if len(records) != 1:
            raise ValueError(f"GenBank file must contain exactly one record: {path}")
        record_features = None
        if expected_features is not None:
            record_features = expected_features.loc[
                expected_features["id"].astype(str).eq(str(expected["id"]))
            ].reset_index(drop=True)
            if record_features.empty:
                raise ValueError(f"GenBank feature table has no rows for {expected['id']}")
        _validate_record_against_manifest_row(records[0], expected, expected_features=record_features)
    return {"status": "pass", "row_count": int(len(manifest)), "genbank_dir_path": str(genbank_dir)}


def _validate_feature_table_identity(manifest: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    missing_columns = sorted(set(GENBANK_FEATURE_COLUMNS).difference(feature_table.columns))
    if missing_columns:
        raise ValueError("GenBank feature table missing required columns: " + ", ".join(missing_columns))
    if feature_table.duplicated(subset=["id", "feature_id"]).any():
        raise ValueError("GenBank feature table contains duplicate candidate and feature IDs")
    expected_ids = set(manifest["id"].astype(str))
    observed_ids = set(feature_table["id"].astype(str))
    if observed_ids != expected_ids:
        missing = sorted(expected_ids.difference(observed_ids))
        extra = sorted(observed_ids.difference(expected_ids))
        raise ValueError(
            f"GenBank feature table candidate ID parity mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    manifest_by_id = manifest.set_index(manifest["id"].astype(str), drop=False)
    for column in ("batch_id", "campaign_slug", "synthesis_name"):
        expected_by_id = manifest_by_id[column].astype(str).to_dict()
        expected_values = feature_table["id"].astype(str).map(expected_by_id)
        observed_values = feature_table[column].astype(str)
        mismatches = feature_table.loc[observed_values.ne(expected_values), ["id", column]]
        if not mismatches.empty:
            raise ValueError(f"GenBank feature table {column} mismatch: {mismatches.head(5).to_dict('records')}")
