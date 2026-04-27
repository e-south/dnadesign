"""
Read-only planner and explicit writer for the Reader SFXI pDual DenseGen archive port.

This is intentionally a one-off migration helper. It does not mutate Reader inputs,
the archived DenseGen dataset, or any existing modern DenseGen dataset. The default
CLI mode is a dry run; dataset creation requires --write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.contracts import compute_id
from dnadesign.usr.src.registry import arrow_type_from_str, load_registry, registry_entry
from dnadesign.usr.src.storage.parquet import now_utc

DEFAULT_OUTPUT_DATASET = "usr_sfxi_pdual10_densegen_promoters"
DEFAULT_ARCHIVE_DATASET = "archived/60bp_dual_promoter_cpxR_LexA"
MODERN_DENSEGEN_SCHEMA_VERSION = "2.9"
PORT_RUN_ID = "reader_sfxi_pdual10_archive_port"
ARCHIVE_SOURCE_DATASET = "archived/60bp_dual_promoter_cpxR_LexA"
TARGET_DENSEGEN_PLAN = "ethanol_ciprofloxacin"
TARGET_DENSEGEN_INPUT_MODE = "plan_pool"
TARGET_DENSEGEN_INPUT_NAME = "plan_pool__ethanol_ciprofloxacin"
PROMOTER_CONSTRAINT_NAME = "sigma70_core"
CANONICAL_SIGMA70_UPSTREAM_35_VARIANTS = {
    "TTGACA": "f",
    "TAGACA": "e",
    "TTTACA": "d",
    "TTGTGA": "c",
    "CTGACA": "b",
}
CANONICAL_SIGMA70_DOWNSTREAM_10_VARIANTS = {
    "TATAAT": "consensus",
}
ARCHIVE_REGULATOR_TO_MODERN = {
    "cpxr": "cpxR_MANWWHTTTAM",
    "lexa": "lexA_CTGTATAWAWWHACA",
}
TARGET_REQUIRED_REGULATORS = tuple(ARCHIVE_REGULATOR_TO_MODERN.values())
SIGMA70_DETAIL_RE = re.compile(r"^sigma70_(?P<variant>[A-Za-z0-9]+)_(?P<role>upstream|downstream)$", re.IGNORECASE)
MODERN_TFBS_DETAIL_FIELDS = (
    "part_kind",
    "role",
    "constraint_name",
    "sequence",
    "core_sequence",
    "variant_id",
    "spacer_length",
    "placement_index",
    "part_index",
    "regulator",
    "motif_id",
    "tfbs_id",
    "orientation",
    "offset",
    "offset_raw",
    "length",
    "end",
    "pad_left",
    "site_id",
    "source",
    "score_best_hit_raw",
    "score_theoretical_max",
    "score_relative_to_theoretical_max",
    "rank_among_mined_positive",
    "rank_among_selected",
    "tier",
    "selection_policy",
    "nearest_selected_similarity",
    "nearest_selected_distance",
    "nearest_selected_distance_norm",
    "matched_start",
    "matched_stop",
    "matched_strand",
)
MODERN_DENSEGEN_COLUMNS = (
    "densegen__schema_version",
    "densegen__created_at",
    "densegen__run_id",
    "densegen__length",
    "densegen__plan",
    "densegen__input_name",
    "densegen__input_mode",
    "densegen__input_pwm_ids",
    "densegen__used_tfbs",
    "densegen__used_tfbs_detail",
    "densegen__used_tf_counts",
    "densegen__library_unique_tf_count",
    "densegen__library_unique_tfbs_count",
    "densegen__covers_all_tfs_in_solution",
    "densegen__required_regulators",
    "densegen__min_count_by_regulator",
    "densegen__compression_ratio",
    "densegen__sampling_library_hash",
    "densegen__sampling_library_index",
    "densegen__pad_used",
    "densegen__pad_bases",
    "densegen__pad_end",
    "densegen__pad_literal",
    "densegen__sequence_validation",
    "densegen__gc_total",
    "densegen__gc_core",
)
IUPAC_DNA = set("ACGTRYSWKMBDHVN")
STRICT_DNA = set("ACGT")
_REVCOMP_TABLE = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class ReaderObservation:
    experiment: str
    metadata_path: str
    sheet: str
    excel_row: int
    design_id: str
    sequence: str

    @property
    def is_pdual_evidence(self) -> bool:
        return "pdual" in self.experiment.lower() and self.design_id.lower().startswith("pdual-10-")


@dataclass(frozen=True)
class ExcludedCandidate:
    design_id: str
    sequence: str
    reason: str
    experiment: str


@dataclass(frozen=True)
class PortCandidate:
    design_id: str
    sequence: str
    reader_experiments: tuple[str, ...]
    archive_row: dict[str, Any]

    @property
    def id(self) -> str:
        return compute_id("dna", self.sequence)

    @property
    def archive_id(self) -> str:
        return str(self.archive_row["id"])


@dataclass(frozen=True)
class PortPlan:
    included: tuple[PortCandidate, ...]
    excluded: tuple[ExcludedCandidate, ...]
    observations: tuple[ReaderObservation, ...]
    blank_sequence_rows: int
    archive_records: str
    reader_root: str

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for row in self.excluded:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
        return {
            "reader_root": self.reader_root,
            "archive_records": self.archive_records,
            "observations": len(self.observations),
            "blank_sequence_rows": self.blank_sequence_rows,
            "included": len(self.included),
            "excluded_by_reason": reasons,
            "included_design_ids": [row.design_id for row in self.included],
        }


@dataclass(frozen=True)
class WriteResult:
    dataset: str
    dataset_dir: str
    rows_written: int
    densegen_overlay_rows: int
    label_overlay_rows: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_reader_root() -> Path:
    return _repo_root().parent / "reader"


def _default_usr_root() -> Path:
    return _repo_root() / "src" / "dnadesign" / "usr" / "datasets"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (list, tuple)) or hasattr(result, "tolist"):
        return False
    return bool(result)


def normalize_sequence_value(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    sequence = re.sub(r"[^A-Za-z]", "", text).upper().replace("U", "T")
    if not sequence:
        return None
    return sequence


def _valid_dna(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= IUPAC_DNA


def _strict_dna(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= STRICT_DNA


def _natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def discover_sfxi_experiments(reader_root: Path) -> tuple[Path, ...]:
    experiments_root = reader_root / "experiments"
    discovered: set[Path] = set()
    for experiment in experiments_root.glob("20*/*"):
        if not experiment.is_dir():
            continue
        name = experiment.name.lower()
        if "sfxi" in name:
            discovered.add(experiment)
            continue
        config = experiment / "config.yaml"
        if config.exists():
            text = config.read_text(encoding="utf-8", errors="ignore").lower()
            if "logic/sfxi_screen" in text or "sfxi" in text:
                discovered.add(experiment)
    return tuple(sorted(discovered))


def _candidate_sequence_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    named = [str(column) for column in frame.columns if "sequence" in str(column).lower()]
    if named:
        return tuple(named)
    candidates: list[str] = []
    for column in frame.columns:
        values = frame[column].dropna().head(20)
        if any((seq := normalize_sequence_value(value)) and len(seq) >= 20 and _valid_dna(seq) for value in values):
            candidates.append(str(column))
    return tuple(candidates)


def load_reader_observations(reader_root: Path) -> tuple[tuple[ReaderObservation, ...], int]:
    observations: list[ReaderObservation] = []
    blank_sequence_rows = 0
    experiments_root = reader_root / "experiments"
    for experiment in discover_sfxi_experiments(reader_root):
        metadata = experiment / "inputs" / "metadata.xlsx"
        if not metadata.exists():
            continue
        workbook = pd.ExcelFile(metadata)
        experiment_id = experiment.relative_to(experiments_root).as_posix()
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(metadata, sheet_name=sheet)
            for column in _candidate_sequence_columns(frame):
                for row_index, value in frame[column].items():
                    sequence = normalize_sequence_value(value)
                    if sequence is None:
                        blank_sequence_rows += 1
                        continue
                    design_id = ""
                    if "design_id" in frame.columns and not _is_missing(frame.at[row_index, "design_id"]):
                        design_id = str(frame.at[row_index, "design_id"]).strip()
                    if not design_id or not _valid_dna(sequence):
                        continue
                    observations.append(
                        ReaderObservation(
                            experiment=experiment_id,
                            metadata_path=str(metadata),
                            sheet=str(sheet),
                            excel_row=int(row_index) + 2,
                            design_id=design_id,
                            sequence=sequence,
                        )
                    )
    observations.sort(key=lambda row: (_natural_key(row.design_id), row.experiment, row.sequence))
    return tuple(observations), blank_sequence_rows


def _archive_records_by_sequence(archive_records: Path) -> dict[str, dict[str, Any]]:
    archive = pq.read_table(archive_records).to_pandas()
    records: dict[str, dict[str, Any]] = {}
    for row in archive.to_dict(orient="records"):
        sequence = normalize_sequence_value(row.get("sequence"))
        if sequence is None:
            continue
        if sequence in records:
            raise ValueError(f"Archive has duplicate sequence rows for sequence {sequence!r}; refusing ambiguous port.")
        records[sequence] = row
    return records


def build_port_plan(*, reader_root: Path, archive_records: Path) -> PortPlan:
    observations, blank_sequence_rows = load_reader_observations(reader_root)
    archive_by_sequence = _archive_records_by_sequence(archive_records)
    excluded: list[ExcludedCandidate] = []
    pdual_archive_observations: list[ReaderObservation] = []

    for observation in observations:
        if not observation.is_pdual_evidence:
            excluded.append(
                ExcludedCandidate(
                    design_id=observation.design_id,
                    sequence=observation.sequence,
                    reason="non_pdual_reader_evidence",
                    experiment=observation.experiment,
                )
            )
            continue
        if len(observation.sequence) != 60:
            excluded.append(
                ExcludedCandidate(
                    design_id=observation.design_id,
                    sequence=observation.sequence,
                    reason="not_60bp",
                    experiment=observation.experiment,
                )
            )
            continue
        if not _strict_dna(observation.sequence):
            excluded.append(
                ExcludedCandidate(
                    design_id=observation.design_id,
                    sequence=observation.sequence,
                    reason="non_strict_dna",
                    experiment=observation.experiment,
                )
            )
            continue
        if observation.sequence not in archive_by_sequence:
            excluded.append(
                ExcludedCandidate(
                    design_id=observation.design_id,
                    sequence=observation.sequence,
                    reason="no_archive_densegen_match",
                    experiment=observation.experiment,
                )
            )
            continue
        pdual_archive_observations.append(observation)

    sequence_to_designs: dict[str, set[str]] = {}
    design_to_sequences: dict[str, set[str]] = {}
    for observation in pdual_archive_observations:
        sequence_to_designs.setdefault(observation.sequence, set()).add(observation.design_id)
        design_to_sequences.setdefault(observation.design_id, set()).add(observation.sequence)

    ambiguous_sequences: set[str] = set()
    for sequence, design_ids in sequence_to_designs.items():
        if len(design_ids) > 1:
            ambiguous_sequences.add(sequence)
    for design_id, sequences in design_to_sequences.items():
        if len(sequences) > 1:
            ambiguous_sequences.update(sequences)
    ambiguous_designs = {
        design_id for design_id, sequences in design_to_sequences.items() if len(sequences & ambiguous_sequences) > 0
    }

    included_by_sequence: dict[str, PortCandidate] = {}
    for observation in pdual_archive_observations:
        if observation.sequence in ambiguous_sequences or observation.design_id in ambiguous_designs:
            excluded.append(
                ExcludedCandidate(
                    design_id=observation.design_id,
                    sequence=observation.sequence,
                    reason="ambiguous_pdual_design_or_sequence",
                    experiment=observation.experiment,
                )
            )
            continue
        if observation.sequence not in included_by_sequence:
            matching = [row for row in pdual_archive_observations if row.sequence == observation.sequence]
            included_by_sequence[observation.sequence] = PortCandidate(
                design_id=observation.design_id,
                sequence=observation.sequence,
                reader_experiments=tuple(sorted({row.experiment for row in matching})),
                archive_row=archive_by_sequence[observation.sequence],
            )

    included = tuple(sorted(included_by_sequence.values(), key=lambda row: _natural_key(row.design_id)))
    excluded.sort(key=lambda row: (row.reason, _natural_key(row.design_id), row.experiment, row.sequence))
    return PortPlan(
        included=included,
        excluded=tuple(excluded),
        observations=observations,
        blank_sequence_rows=blank_sequence_rows,
        archive_records=str(archive_records),
        reader_root=str(reader_root),
    )


def _listify(value: object) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _int_or_none(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return bool(value)


def _tfbs_sequence(detail: dict[str, Any]) -> str:
    if detail.get("tfbs"):
        return str(detail["tfbs"]).strip().upper()
    if detail.get("sequence"):
        return str(detail["sequence"]).strip().upper()
    return ""


def _revcomp(sequence: str) -> str:
    return str(sequence).upper().translate(_REVCOMP_TABLE)[::-1]


def _stable_short_hash(payload: dict[str, Any], *, prefix: str) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _canonical_regulator(value: object) -> str | None:
    raw = str(value or "").strip()
    if raw == "":
        return None
    lower = raw.lower()
    if lower in ARCHIVE_REGULATOR_TO_MODERN:
        return ARCHIVE_REGULATOR_TO_MODERN[lower]
    for modern in TARGET_REQUIRED_REGULATORS:
        if lower == modern.lower():
            return modern
    return None


def _normalize_orientation(value: object) -> str:
    orientation = str(value or "").strip().lower()
    if orientation not in {"fwd", "rev"}:
        raise ValueError(f"Archive TFBS detail has unsupported orientation {orientation!r}.")
    return orientation


def _placed_sequence(sequence: str, orientation: str) -> str:
    return sequence if orientation == "fwd" else _revcomp(sequence)


def _assert_detail_matches_sequence(
    *, candidate: PortCandidate, detail: dict[str, Any], sequence: str, offset: int
) -> None:
    orientation = _normalize_orientation(detail.get("orientation"))
    placed = _placed_sequence(sequence, orientation)
    observed = candidate.sequence[offset : offset + len(placed)]
    if observed != placed:
        tf = str(detail.get("tf") or detail.get("regulator") or "").strip()
        raise ValueError(
            "Archive DenseGen detail does not match the selected final sequence "
            f"(design_id={candidate.design_id}, tf={tf!r}, offset={offset}, "
            f"expected={placed!r}, observed={observed!r})."
        )


def _blank_modern_detail() -> dict[str, Any]:
    return {field: None for field in MODERN_TFBS_DETAIL_FIELDS}


def _detail_identity(*, kind: str, regulator: str | None, sequence: str, orientation: str, offset: int) -> str:
    return _stable_short_hash(
        {
            "kind": kind,
            "regulator": regulator,
            "sequence": sequence,
            "orientation": orientation,
            "offset": offset,
            "source": ARCHIVE_SOURCE_DATASET,
        },
        prefix=kind,
    )


def _modern_tfbs_detail(detail: object, *, candidate: PortCandidate, part_index: int, pad_left: int) -> dict[str, Any]:
    raw = dict(detail or {}) if isinstance(detail, dict) else {}
    sequence = _tfbs_sequence(raw)
    offset = _int_or_none(raw.get("offset"))
    length = len(sequence) if sequence else _int_or_none(raw.get("length"))
    if not sequence or offset is None or length is None:
        raise ValueError(f"Archive TFBS detail is missing sequence/offset fields for {candidate.design_id}.")
    regulator = _canonical_regulator(raw.get("tf") or raw.get("regulator"))
    if regulator is None:
        raise ValueError(
            "Archive TFBS detail uses an unsupported regulator for the target ethanol/ciprofloxacin plan "
            f"(design_id={candidate.design_id}, tf={raw.get('tf')!r})."
        )
    orientation = _normalize_orientation(raw.get("orientation"))
    _assert_detail_matches_sequence(candidate=candidate, detail=raw, sequence=sequence, offset=offset)
    tfbs_id = _detail_identity(
        kind="tfbs",
        regulator=regulator,
        sequence=sequence,
        orientation=orientation,
        offset=offset,
    )
    out = _blank_modern_detail()
    out.update(
        {
            "part_kind": "tfbs",
            "sequence": sequence or None,
            "core_sequence": sequence or None,
            "regulator": regulator,
            "motif_id": regulator,
            "tfbs_id": tfbs_id,
            "orientation": orientation,
            "offset": offset,
            "offset_raw": offset,
            "length": length,
            "end": offset + length,
            "pad_left": pad_left,
            "site_id": tfbs_id,
            "source": ARCHIVE_SOURCE_DATASET,
            "part_index": part_index,
            "selection_policy": "archive_densegen_port",
        }
    )
    return out


def _modern_fixed_element_detail(
    detail: dict[str, Any],
    *,
    candidate: PortCandidate,
    role: str,
    variant_id: str,
    spacer_length: int,
    pad_left: int,
    placement_index: int = 0,
) -> dict[str, Any]:
    sequence = _tfbs_sequence(detail)
    offset = _int_or_none(detail.get("offset"))
    if not sequence or offset is None:
        raise ValueError(f"Archive sigma70 detail is missing sequence/offset fields for {candidate.design_id}.")
    orientation = _normalize_orientation(detail.get("orientation"))
    if orientation != "fwd":
        raise ValueError(
            "Archive sigma70 fixed-element details must be parent-orientation forward placements "
            f"(design_id={candidate.design_id}, role={role}, orientation={orientation})."
        )
    _assert_detail_matches_sequence(candidate=candidate, detail=detail, sequence=sequence, offset=offset)
    length = len(sequence)
    out = _blank_modern_detail()
    out.update(
        {
            "part_kind": "fixed_element",
            "role": role,
            "constraint_name": PROMOTER_CONSTRAINT_NAME,
            "sequence": sequence,
            "core_sequence": sequence,
            "variant_id": variant_id,
            "spacer_length": spacer_length,
            "placement_index": placement_index,
            "orientation": orientation,
            "offset": offset,
            "offset_raw": offset,
            "length": length,
            "end": offset + length,
            "pad_left": pad_left,
            "source": ARCHIVE_SOURCE_DATASET,
            "selection_policy": "archive_densegen_port",
        }
    )
    return out


def _archive_detail_tf(detail: dict[str, Any]) -> str:
    return str(detail.get("tf") or detail.get("regulator") or "").strip()


def _modern_parts_detail(candidate: PortCandidate, *, pad_left: int) -> list[dict[str, Any]]:
    tfbs_details: list[dict[str, Any]] = []
    fixed_by_role: dict[str, tuple[str, dict[str, Any]]] = {}
    for raw_detail in _listify(candidate.archive_row.get("densegen__used_tfbs_detail")):
        if not isinstance(raw_detail, dict):
            raise ValueError(f"Archive used_tfbs_detail contains non-dict entries for {candidate.design_id}.")
        raw = dict(raw_detail)
        tf_name = _archive_detail_tf(raw)
        match = SIGMA70_DETAIL_RE.fullmatch(tf_name)
        if match:
            role = match.group("role").lower()
            if role in fixed_by_role:
                raise ValueError(f"Archive row has multiple sigma70 {role} details for {candidate.design_id}.")
            fixed_by_role[role] = (tf_name, raw)
            continue
        if _canonical_regulator(tf_name) is None:
            raise ValueError(
                "Archive row contains a TFBS regulator outside the target cpxR/LexA ethanol/ciprofloxacin port "
                f"(design_id={candidate.design_id}, tf={tf_name!r})."
            )
        tfbs_details.append(
            _modern_tfbs_detail(raw, candidate=candidate, part_index=len(tfbs_details), pad_left=pad_left)
        )

    counts = _tfbs_counts(tfbs_details)
    missing = [regulator for regulator in TARGET_REQUIRED_REGULATORS if counts.get(regulator, 0) < 1]
    if missing:
        raise ValueError(
            "Reader SFXI pDual archive port requires both cpxR and LexA evidence for the combined plan "
            f"(design_id={candidate.design_id}, missing={missing})."
        )
    if set(fixed_by_role) != {"upstream", "downstream"}:
        raise ValueError(
            "Reader SFXI pDual archive port requires one sigma70 upstream and one sigma70 downstream fixed element "
            f"(design_id={candidate.design_id}, roles={sorted(fixed_by_role)})."
        )
    _upstream_archive_label, upstream = fixed_by_role["upstream"]
    _downstream_archive_label, downstream = fixed_by_role["downstream"]
    upstream_offset = _int_or_none(upstream.get("offset"))
    downstream_offset = _int_or_none(downstream.get("offset"))
    upstream_sequence = _tfbs_sequence(upstream)
    downstream_sequence = _tfbs_sequence(downstream)
    if upstream_offset is None or downstream_offset is None:
        raise ValueError(f"Archive sigma70 fixed elements are missing offsets for {candidate.design_id}.")
    spacer_length = downstream_offset - (upstream_offset + len(upstream_sequence))
    if spacer_length < 0:
        raise ValueError(f"Archive sigma70 downstream site overlaps upstream site for {candidate.design_id}.")
    fixed_details = [
        _modern_fixed_element_detail(
            upstream,
            candidate=candidate,
            role="upstream",
            variant_id=_sigma70_variant_id(upstream_sequence, role="upstream"),
            spacer_length=spacer_length,
            pad_left=pad_left,
        ),
        _modern_fixed_element_detail(
            downstream,
            candidate=candidate,
            role="downstream",
            variant_id=_sigma70_variant_id(downstream_sequence, role="downstream"),
            spacer_length=spacer_length,
            pad_left=pad_left,
        ),
    ]
    return [*tfbs_details, *fixed_details]


def _sigma70_variant_id(sequence: str, *, role: str) -> str:
    literal = str(sequence or "").strip().upper()
    if role == "upstream":
        return CANONICAL_SIGMA70_UPSTREAM_35_VARIANTS.get(literal, literal)
    if role == "downstream":
        return CANONICAL_SIGMA70_DOWNSTREAM_10_VARIANTS.get(literal, literal)
    raise ValueError(f"Unsupported sigma70 fixed-element role {role!r}.")


def _tfbs_counts(details: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detail in details:
        if str(detail.get("part_kind") or "").lower() != "tfbs":
            continue
        regulator = str(detail.get("regulator") or "").strip()
        if regulator:
            counts[regulator] = counts.get(regulator, 0) + 1
    return counts


def _modern_used_tf_counts(details: list[dict[str, Any]]) -> list[dict[str, int]]:
    counts = _tfbs_counts(details)
    return [{"tf": tf, "count": counts[tf]} for tf in sorted(counts)]


def _min_count_by_regulator() -> list[dict[str, int]]:
    return [{"tf": regulator, "min_count": 1} for regulator in TARGET_REQUIRED_REGULATORS]


def _gc_fraction(sequence: str) -> float:
    return float(sum(1 for base in sequence if base in {"G", "C"})) / float(len(sequence))


def _pad_literal(sequence: str, *, pad_used: bool, pad_bases: int | None, pad_end: object) -> str | None:
    if not pad_used or not pad_bases:
        return None
    end = str(pad_end or "").strip().lower()
    if end in {"5prime", "left"}:
        return sequence[:pad_bases]
    if end in {"3prime", "right"}:
        return sequence[-pad_bases:]
    return None


def _core_sequence(sequence: str, *, pad_used: bool, pad_bases: int | None, pad_end: object) -> str:
    if not pad_used or not pad_bases:
        return sequence
    end = str(pad_end or "").strip().lower()
    if end in {"5prime", "left"}:
        return sequence[pad_bases:] or sequence
    if end in {"3prime", "right"}:
        return sequence[:-pad_bases] or sequence
    return sequence


def _sampling_library_hash(candidate: PortCandidate, details: list[dict[str, Any]]) -> str:
    return _stable_short_hash(
        {
            "source_dataset": ARCHIVE_SOURCE_DATASET,
            "archive_id": candidate.archive_id,
            "target_plan": TARGET_DENSEGEN_PLAN,
            "required_regulators": TARGET_REQUIRED_REGULATORS,
            "tfbs": [
                {
                    "regulator": detail.get("regulator"),
                    "sequence": detail.get("sequence"),
                    "orientation": detail.get("orientation"),
                }
                for detail in details
                if str(detail.get("part_kind") or "").lower() == "tfbs"
            ],
            "fixed_elements": [
                {
                    "role": detail.get("role"),
                    "sequence": detail.get("sequence"),
                    "variant_id": detail.get("variant_id"),
                    "spacer_length": detail.get("spacer_length"),
                }
                for detail in details
                if str(detail.get("part_kind") or "").lower() == "fixed_element"
            ],
        },
        prefix="archive_library",
    )


def _archive_row_to_modern_densegen(candidate: PortCandidate) -> dict[str, Any]:
    row = candidate.archive_row
    sequence = candidate.sequence
    pad_used = _bool_or_none(row.get("densegen__gap_fill_used")) or False
    pad_bases = _int_or_none(row.get("densegen__gap_fill_bases")) if pad_used else None
    pad_end = row.get("densegen__gap_fill_end") if pad_used else None
    pad_left = int(pad_bases or 0) if pad_used and str(pad_end).lower() in {"5prime", "left"} else 0
    details = _modern_parts_detail(candidate, pad_left=pad_left)
    used_counts = _modern_used_tf_counts(details)
    used_tfbs = [
        f"{detail['regulator']}:{detail['sequence']}"
        for detail in details
        if str(detail.get("part_kind") or "").lower() == "tfbs"
    ]
    gc = _gc_fraction(sequence)
    core_sequence = _core_sequence(sequence, pad_used=pad_used, pad_bases=pad_bases, pad_end=pad_end)
    return {
        "id": candidate.id,
        "densegen__schema_version": MODERN_DENSEGEN_SCHEMA_VERSION,
        "densegen__created_at": str(row.get("created_at") or ""),
        "densegen__run_id": PORT_RUN_ID,
        "densegen__length": len(sequence),
        "densegen__plan": TARGET_DENSEGEN_PLAN,
        "densegen__input_name": TARGET_DENSEGEN_INPUT_NAME,
        "densegen__input_mode": TARGET_DENSEGEN_INPUT_MODE,
        "densegen__input_pwm_ids": [],
        "densegen__used_tfbs": used_tfbs,
        "densegen__used_tfbs_detail": details,
        "densegen__used_tf_counts": used_counts,
        "densegen__library_unique_tf_count": len({detail["regulator"] for detail in details if detail["regulator"]}),
        "densegen__library_unique_tfbs_count": len(set(used_tfbs)),
        "densegen__covers_all_tfs_in_solution": True,
        "densegen__required_regulators": list(TARGET_REQUIRED_REGULATORS),
        "densegen__min_count_by_regulator": _min_count_by_regulator(),
        "densegen__compression_ratio": _float_or_none(row.get("densegen__compression_ratio")),
        "densegen__sampling_library_hash": _sampling_library_hash(candidate, details),
        "densegen__sampling_library_index": None,
        "densegen__pad_used": pad_used,
        "densegen__pad_bases": pad_bases,
        "densegen__pad_end": pad_end,
        "densegen__pad_literal": _pad_literal(sequence, pad_used=pad_used, pad_bases=pad_bases, pad_end=pad_end),
        "densegen__sequence_validation": {"validation_passed": True, "violations": []},
        "densegen__gc_total": gc,
        "densegen__gc_core": _gc_fraction(core_sequence),
    }


def modern_densegen_overlay_frame(candidates: Iterable[PortCandidate]) -> pd.DataFrame:
    rows = [_archive_row_to_modern_densegen(candidate) for candidate in candidates]
    library_indexes = {
        library_hash: index
        for index, library_hash in enumerate(
            sorted({str(row["densegen__sampling_library_hash"]) for row in rows}),
            start=1,
        )
    }
    for row in rows:
        row["densegen__sampling_library_index"] = library_indexes[str(row["densegen__sampling_library_hash"])]
    return pd.DataFrame(rows, columns=("id", *MODERN_DENSEGEN_COLUMNS), dtype=object)


def base_records_frame(candidates: Iterable[PortCandidate]) -> pd.DataFrame:
    rows = []
    created_at = now_utc()
    for candidate in candidates:
        rows.append(
            {
                "id": compute_id("dna", candidate.sequence),
                "bio_type": "dna",
                "sequence": candidate.sequence,
                "alphabet": "dna_4",
                "source": f"{ARCHIVE_SOURCE_DATASET};reader_sfxi_pdual",
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)


def usr_label_overlay_frame(candidates: Iterable[PortCandidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        short_alias = candidate.design_id.replace("pDual-10-", "")
        aliases = [
            short_alias,
            f"archive_id:{candidate.archive_id}",
            *(f"reader_experiment:{experiment}" for experiment in candidate.reader_experiments),
        ]
        rows.append(
            {
                "id": candidate.id,
                "usr_label__primary": candidate.design_id,
                "usr_label__aliases": sorted(set(aliases)),
            }
        )
    return pd.DataFrame(rows)


def _overlay_schema_from_registry(usr_root: Path, namespace: str, columns: Iterable[str]) -> pa.Schema:
    entry = registry_entry(load_registry(usr_root, required=True), namespace)
    allowed = {column.name: arrow_type_from_str(column.type) for column in entry.columns}
    fields = [pa.field("id", pa.string())]
    for column in columns:
        if column == "id":
            continue
        fields.append(pa.field(column, allowed[column]))
    return pa.schema(fields)


def _table_from_frame(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    def _clean(value: object, target_type: pa.DataType) -> object:
        if _is_missing(value):
            return None
        if pa.types.is_string(target_type):
            return str(value)
        if pa.types.is_integer(target_type):
            return _int_or_none(value)
        if pa.types.is_floating(target_type):
            return _float_or_none(value)
        if pa.types.is_boolean(target_type):
            return _bool_or_none(value)
        if pa.types.is_list(target_type) or pa.types.is_large_list(target_type):
            return [_clean(item, target_type.value_type) for item in _listify(value)]
        if pa.types.is_struct(target_type):
            if not isinstance(value, dict):
                return None
            return {field.name: _clean(value.get(field.name), field.type) for field in target_type}
        return value

    raw_rows = frame.to_dict(orient="records")
    rows = [{field.name: _clean(row.get(field.name), field.type) for field in schema} for row in raw_rows]
    return pa.Table.from_pylist(rows, schema=schema)


def _validate_included_count(plan: PortPlan, expected_count: int | None) -> None:
    if expected_count is not None and len(plan.included) != expected_count:
        raise ValueError(f"Expected {expected_count} included rows, found {len(plan.included)}.")


def write_port_dataset(
    plan: PortPlan,
    *,
    usr_root: Path,
    output_dataset: str = DEFAULT_OUTPUT_DATASET,
    expected_count: int | None = None,
) -> WriteResult:
    _validate_included_count(plan, expected_count)
    dataset = Dataset(usr_root, output_dataset)
    if dataset.dir.exists():
        raise FileExistsError(f"Output dataset already exists: {dataset.dir}")

    base_rows = base_records_frame(plan.included)
    densegen_frame = modern_densegen_overlay_frame(plan.included)
    labels_frame = usr_label_overlay_frame(plan.included)
    densegen_schema = _overlay_schema_from_registry(usr_root, "densegen", densegen_frame.columns)
    label_schema = _overlay_schema_from_registry(usr_root, "usr_label", labels_frame.columns)
    densegen_table = _table_from_frame(densegen_frame, densegen_schema)
    label_table = _table_from_frame(labels_frame, label_schema)

    with dataset.write_session() as session:
        session.init(
            source=f"{ARCHIVE_SOURCE_DATASET} + Reader SFXI pDual metadata",
            notes="Curated port of archive-backed Reader SFXI pDual promoter sequences.",
        )
        rows_written = session.import_rows(base_rows, source=f"{ARCHIVE_SOURCE_DATASET};reader_sfxi_pdual")
        densegen_rows = session.write_overlay(
            "densegen",
            densegen_table,
            key="id",
            note="modern DenseGen overlay mapped from archived 60bp dual-promoter dataset",
        )
        label_rows = session.write_overlay(
            "usr_label",
            label_table,
            key="id",
            note="Reader SFXI pDual labels and provenance aliases",
        )
    dataset.validate(strict=True)
    return WriteResult(
        dataset=dataset.name,
        dataset_dir=str(dataset.dir),
        rows_written=int(rows_written),
        densegen_overlay_rows=int(densegen_rows),
        label_overlay_rows=int(label_rows),
    )


def _write_report(plan: PortPlan, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            candidate.__dict__
            | {
                "target_id": candidate.id,
                "archive_id": candidate.archive_id,
            }
            for candidate in plan.included
        ]
    ).to_csv(
        report_dir / "included.csv",
        index=False,
    )
    pd.DataFrame([row.__dict__ for row in plan.excluded]).to_csv(report_dir / "excluded.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(plan.summary(), indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Port curated Reader SFXI pDual archive-backed sequences into a new modern USR dataset."
    )
    parser.add_argument("--reader-root", type=Path, default=_default_reader_root())
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--archive-records", type=Path, default=None)
    parser.add_argument("--output-dataset", default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--expected-count", type=int, default=23)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--write", action="store_true", help="Actually create the output USR dataset. Default is dry-run."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_records = args.archive_records or (args.usr_root / DEFAULT_ARCHIVE_DATASET / "records.parquet")
    plan = build_port_plan(reader_root=args.reader_root, archive_records=archive_records)
    _validate_included_count(plan, args.expected_count)
    if args.report_dir is not None:
        _write_report(plan, args.report_dir)
    payload: dict[str, Any] = {"plan": plan.summary(), "write": bool(args.write)}
    if args.write:
        result = write_port_dataset(
            plan,
            usr_root=args.usr_root,
            output_dataset=args.output_dataset,
            expected_count=args.expected_count,
        )
        payload["result"] = result.__dict__
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
