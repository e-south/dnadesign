"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/regulondb/functional_annotations.py

Projects BioCyc regulator GO annotations onto RegulonDB promoter USR sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

REGULATOR_GO_TERMS_SIDECAR = "regulator_go_terms.parquet"
PROMOTER_REGULATOR_GO_TERMS_SIDECAR = "promoter_regulator_go_terms.parquet"
REGULATOR_GO_COVERAGE_SIDECAR = "regulator_go_coverage.parquet"

GO_ID_PATTERN = re.compile(r"^GO:\d{7}$")

REQUIRED_REGULATORY_INTERACTION_COLUMNS = (
    "usr_id",
    "promoter_id",
    "regulator_id",
    "regulator_name",
    "regulatory_interaction_id",
)
REQUIRED_TERMS_COLUMNS = (
    "regulator_id",
    "regulator_name",
    "regulator_gene_name",
    "gene_symbol",
    "go_aspect",
    "go_id",
    "go_name",
    "go_namespace",
    "source_column",
    "source_route",
    "identity_source_id",
    "biocyc_kb_version",
    "smarttable_id",
)
REQUIRED_COVERAGE_COLUMNS = (
    "regulator_id",
    "regulator_name",
    "regulator_gene_name",
    "gene_symbol_count",
    "matched_go_term_count",
    "mapping_status",
    "identity_source_id",
    "biocyc_kb_version",
    "smarttable_id",
)

REGULATOR_GO_TERMS_SCHEMA = pa.schema(
    [
        pa.field("regulator_id", pa.string()),
        pa.field("regulator_name", pa.string()),
        pa.field("regulator_gene_name", pa.string()),
        pa.field("gene_symbol", pa.string()),
        pa.field("go_aspect", pa.string()),
        pa.field("go_id", pa.string()),
        pa.field("go_name", pa.string()),
        pa.field("go_namespace", pa.string()),
        pa.field("source_column", pa.string()),
        pa.field("source_route", pa.string()),
        pa.field("identity_source_id", pa.string()),
        pa.field("biocyc_kb_version", pa.string()),
        pa.field("smarttable_id", pa.string()),
        pa.field("source_terms_sha256", pa.string()),
    ]
)
PROMOTER_REGULATOR_GO_TERMS_SCHEMA = pa.schema(
    [
        pa.field("usr_id", pa.string()),
        pa.field("promoter_id", pa.string()),
        pa.field("regulator_id", pa.string()),
        pa.field("regulator_name", pa.string()),
        pa.field("gene_symbol", pa.string()),
        pa.field("go_aspect", pa.string()),
        pa.field("go_id", pa.string()),
        pa.field("go_name", pa.string()),
        pa.field("go_namespace", pa.string()),
        pa.field("biocyc_kb_version", pa.string()),
        pa.field("smarttable_id", pa.string()),
        pa.field("source_terms_sha256", pa.string()),
    ]
)
REGULATOR_GO_COVERAGE_SCHEMA = pa.schema(
    [
        pa.field("regulator_id", pa.string()),
        pa.field("regulator_name", pa.string()),
        pa.field("regulator_gene_name", pa.string()),
        pa.field("usr_promoter_count", pa.int64()),
        pa.field("source_promoter_count", pa.int64()),
        pa.field("regulatory_interaction_count", pa.int64()),
        pa.field("matched_go_term_count", pa.int64()),
        pa.field("mapping_status", pa.string()),
        pa.field("identity_source_id", pa.string()),
        pa.field("biocyc_kb_version", pa.string()),
        pa.field("smarttable_id", pa.string()),
        pa.field("source_coverage_sha256", pa.string()),
        pa.field("source_terms_sha256", pa.string()),
    ]
)


@dataclass(frozen=True)
class RegulonDBGoProjection:
    regulator_go_terms_rows: list[dict[str, object]]
    promoter_regulator_go_terms_rows: list[dict[str, object]]
    regulator_go_coverage_rows: list[dict[str, object]]
    summary: dict[str, object]


def build_regulondb_go_projection(
    *,
    dataset_root: Path,
    terms_path: Path,
    coverage_path: Path,
    min_covered_regulator_fraction: float = 0.95,
) -> RegulonDBGoProjection:
    """Build promoter/regulator GO sidecar rows from BioCyc SmartTable outputs."""

    dataset_root = Path(dataset_root)
    terms_path = Path(terms_path)
    coverage_path = Path(coverage_path)
    _require_fraction(min_covered_regulator_fraction)
    _require_file(dataset_root / "records.parquet", "RegulonDB USR records")
    regulatory_interactions_path = dataset_root / "_relations" / "regulatory_interactions.parquet"
    _require_file(regulatory_interactions_path, "RegulonDB regulatory interaction sidecar")
    _require_file(terms_path, "BioCyc regulator GO terms")
    _require_file(coverage_path, "BioCyc regulator GO coverage")

    records = pq.read_table(dataset_root / "records.parquet", columns=["id"])
    base_ids = {str(value) for value in records.column("id").to_pylist() if _text(value)}
    interactions_table = pq.read_table(regulatory_interactions_path)
    _require_columns(
        interactions_table.schema.names,
        REQUIRED_REGULATORY_INTERACTION_COLUMNS,
        regulatory_interactions_path,
    )
    interactions = _table_rows(interactions_table)
    _validate_interactions(interactions, base_ids)

    term_rows = _read_tsv_rows(terms_path, required_columns=REQUIRED_TERMS_COLUMNS)
    coverage_rows = _read_tsv_rows(coverage_path, required_columns=REQUIRED_COVERAGE_COLUMNS)
    terms_sha256 = _sha256(terms_path)
    coverage_sha256 = _sha256(coverage_path)
    terms_by_regulator = _terms_by_regulator(term_rows, terms_sha256=terms_sha256)
    coverage_by_regulator = _coverage_by_regulator(coverage_rows)

    regulator_names = _regulator_names(interactions)
    usr_sets = _usr_ids_by_regulator(interactions)
    promoter_sets = _promoters_by_regulator(interactions)
    interaction_counts = _interaction_counts_by_regulator(interactions)
    interacting_regulators = sorted(usr_sets)
    missing_coverage = [
        regulator_id for regulator_id in interacting_regulators if regulator_id not in coverage_by_regulator
    ]
    if missing_coverage:
        raise ValueError(
            "BioCyc coverage table is missing interacting RegulonDB regulator IDs: " + ", ".join(missing_coverage[:10])
        )

    regulator_go_terms_rows = [
        row for regulator_id in interacting_regulators for row in terms_by_regulator.get(regulator_id, [])
    ]
    promoter_regulator_go_terms_rows = _promoter_term_rows(
        interactions,
        terms_by_regulator=terms_by_regulator,
        terms_sha256=terms_sha256,
    )
    regulator_go_coverage_rows = [
        _coverage_output_row(
            regulator_id,
            coverage_by_regulator[regulator_id],
            regulator_names=regulator_names,
            usr_promoter_count=len(usr_sets[regulator_id]),
            source_promoter_count=len(promoter_sets[regulator_id]),
            regulatory_interaction_count=interaction_counts[regulator_id],
            matched_go_term_count=len(terms_by_regulator.get(regulator_id, [])),
            terms_sha256=terms_sha256,
            coverage_sha256=coverage_sha256,
        )
        for regulator_id in interacting_regulators
    ]
    covered_regulators = sum(1 for row in regulator_go_coverage_rows if int(row["matched_go_term_count"]) > 0)
    coverage_fraction = covered_regulators / len(interacting_regulators) if interacting_regulators else 0.0
    if coverage_fraction < min_covered_regulator_fraction:
        raise ValueError(
            "BioCyc GO coverage below required minimum: "
            f"{covered_regulators}/{len(interacting_regulators)} "
            f"({coverage_fraction:.3f} < {min_covered_regulator_fraction:.3f})"
        )

    return RegulonDBGoProjection(
        regulator_go_terms_rows=regulator_go_terms_rows,
        promoter_regulator_go_terms_rows=promoter_regulator_go_terms_rows,
        regulator_go_coverage_rows=regulator_go_coverage_rows,
        summary={
            "dataset": dataset_root.name,
            "interacting_regulator_count": len(interacting_regulators),
            "covered_regulator_count": covered_regulators,
            "coverage_fraction": coverage_fraction,
            "regulator_go_term_rows": len(regulator_go_terms_rows),
            "promoter_regulator_go_term_rows": len(promoter_regulator_go_terms_rows),
            "regulator_go_coverage_rows": len(regulator_go_coverage_rows),
            "uncovered_regulators": [
                {
                    "regulator_id": row["regulator_id"],
                    "regulator_name": row["regulator_name"],
                    "mapping_status": row["mapping_status"],
                }
                for row in regulator_go_coverage_rows
                if int(row["matched_go_term_count"]) == 0
            ],
            "terms_path": str(terms_path),
            "coverage_path": str(coverage_path),
            "terms_sha256": terms_sha256,
            "coverage_sha256": coverage_sha256,
        },
    )


def write_regulondb_go_projection(
    projection: RegulonDBGoProjection,
    *,
    dataset_root: Path,
) -> None:
    """Materialize the BioCyc GO projection as additive USR relation sidecars."""

    relations_dir = Path(dataset_root) / "_relations"
    relations_dir.mkdir(parents=True, exist_ok=True)
    _write_sidecar(
        projection.regulator_go_terms_rows,
        REGULATOR_GO_TERMS_SCHEMA,
        relations_dir / REGULATOR_GO_TERMS_SIDECAR,
    )
    _write_sidecar(
        projection.promoter_regulator_go_terms_rows,
        PROMOTER_REGULATOR_GO_TERMS_SCHEMA,
        relations_dir / PROMOTER_REGULATOR_GO_TERMS_SIDECAR,
    )
    _write_sidecar(
        projection.regulator_go_coverage_rows,
        REGULATOR_GO_COVERAGE_SCHEMA,
        relations_dir / REGULATOR_GO_COVERAGE_SIDECAR,
    )


def _table_rows(table: pa.Table) -> list[dict[str, object]]:
    names = table.schema.names
    columns = {name: table[name].to_pylist() for name in names}
    rows: list[dict[str, object]] = []
    for index in range(table.num_rows):
        rows.append({name: columns[name][index] for name in names})
    return rows


def _validate_interactions(rows: list[dict[str, object]], base_ids: set[str]) -> None:
    if not rows:
        raise ValueError("RegulonDB regulatory_interactions.parquet is empty")
    missing_usr_ids = sorted({_text(row.get("usr_id")) for row in rows if not _text(row.get("usr_id"))})
    if missing_usr_ids:
        raise ValueError("RegulonDB regulatory interactions contain blank usr_id values")
    orphan_usr_ids = sorted({_text(row.get("usr_id")) for row in rows if _text(row.get("usr_id")) not in base_ids})
    if orphan_usr_ids:
        raise ValueError(
            "RegulonDB regulatory interactions contain orphan usr_id values: " + ", ".join(orphan_usr_ids[:10])
        )
    blank_regulators = sum(1 for row in rows if not _text(row.get("regulator_id")))
    if blank_regulators:
        raise ValueError(f"RegulonDB regulatory interactions contain {blank_regulators} blank regulator_id values")


def _read_tsv_rows(path: Path, *, required_columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV file has no header: {path}")
        _require_columns(reader.fieldnames, required_columns, path)
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"TSV file has no data rows: {path}")
    return rows


def _terms_by_regulator(
    rows: list[dict[str, str]],
    *,
    terms_sha256: str,
) -> dict[str, list[dict[str, object]]]:
    by_regulator: dict[str, dict[tuple[str, str, str], dict[str, object]]] = {}
    for row in rows:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        go_id = _require_text(row.get("go_id"), "go_id")
        if not GO_ID_PATTERN.match(go_id):
            raise ValueError(f"BioCyc regulator GO row has invalid GO ID: {go_id!r}")
        key = (str(row.get("gene_symbol") or ""), str(row.get("go_aspect") or ""), go_id)
        by_regulator.setdefault(regulator_id, {})[key] = {
            **{field: row[field] for field in REQUIRED_TERMS_COLUMNS},
            "source_terms_sha256": terms_sha256,
        }
    return {regulator_id: list(rows_by_key.values()) for regulator_id, rows_by_key in by_regulator.items()}


def _coverage_by_regulator(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_regulator: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        by_regulator.setdefault(regulator_id, []).append(row)
    return by_regulator


def _promoter_term_rows(
    interactions: list[dict[str, object]],
    *,
    terms_by_regulator: Mapping[str, list[dict[str, object]]],
    terms_sha256: str,
) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for interaction in interactions:
        usr_id = _require_text(interaction.get("usr_id"), "usr_id")
        promoter_id = _require_text(interaction.get("promoter_id"), "promoter_id")
        regulator_id = _require_text(interaction.get("regulator_id"), "regulator_id")
        regulator_name = _text(interaction.get("regulator_name"))
        for term in terms_by_regulator.get(regulator_id, []):
            key = (
                usr_id,
                promoter_id,
                regulator_id,
                str(term["gene_symbol"]),
                str(term["go_id"]),
            )
            rows_by_key[key] = {
                "usr_id": usr_id,
                "promoter_id": promoter_id,
                "regulator_id": regulator_id,
                "regulator_name": regulator_name or str(term["regulator_name"]),
                "gene_symbol": term["gene_symbol"],
                "go_aspect": term["go_aspect"],
                "go_id": term["go_id"],
                "go_name": term["go_name"],
                "go_namespace": term["go_namespace"],
                "biocyc_kb_version": term["biocyc_kb_version"],
                "smarttable_id": term["smarttable_id"],
                "source_terms_sha256": terms_sha256,
            }
    return list(rows_by_key.values())


def _coverage_output_row(
    regulator_id: str,
    coverage_rows: list[dict[str, str]],
    *,
    regulator_names: Mapping[str, str],
    usr_promoter_count: int,
    source_promoter_count: int,
    regulatory_interaction_count: int,
    matched_go_term_count: int,
    terms_sha256: str,
    coverage_sha256: str,
) -> dict[str, object]:
    preferred = _preferred_coverage_row(coverage_rows)
    status = "matched" if matched_go_term_count > 0 else str(preferred["mapping_status"])
    return {
        "regulator_id": regulator_id,
        "regulator_name": regulator_names.get(regulator_id) or preferred["regulator_name"],
        "regulator_gene_name": preferred["regulator_gene_name"],
        "usr_promoter_count": int(usr_promoter_count),
        "source_promoter_count": int(source_promoter_count),
        "regulatory_interaction_count": int(regulatory_interaction_count),
        "matched_go_term_count": int(matched_go_term_count),
        "mapping_status": status,
        "identity_source_id": preferred["identity_source_id"],
        "biocyc_kb_version": preferred["biocyc_kb_version"],
        "smarttable_id": preferred["smarttable_id"],
        "source_coverage_sha256": coverage_sha256,
        "source_terms_sha256": terms_sha256,
    }


def _preferred_coverage_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return sorted(
        rows,
        key=lambda row: (
            0 if int(row.get("matched_go_term_count") or 0) > 0 else 1,
            str(row.get("mapping_status") or ""),
            str(row.get("regulator_name") or ""),
        ),
    )[0]


def _regulator_names(interactions: list[dict[str, object]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in interactions:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        name = _text(row.get("regulator_name"))
        if name:
            names.setdefault(regulator_id, name)
    return names


def _usr_ids_by_regulator(interactions: list[dict[str, object]]) -> dict[str, set[str]]:
    usr_ids: dict[str, set[str]] = {}
    for row in interactions:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        usr_id = _require_text(row.get("usr_id"), "usr_id")
        usr_ids.setdefault(regulator_id, set()).add(usr_id)
    return usr_ids


def _promoters_by_regulator(interactions: list[dict[str, object]]) -> dict[str, set[str]]:
    promoters: dict[str, set[str]] = {}
    for row in interactions:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        promoter_id = _require_text(row.get("promoter_id"), "promoter_id")
        promoters.setdefault(regulator_id, set()).add(promoter_id)
    return promoters


def _interaction_counts_by_regulator(interactions: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in interactions:
        regulator_id = _require_text(row.get("regulator_id"), "regulator_id")
        counts[regulator_id] = counts.get(regulator_id, 0) + 1
    return counts


def _write_sidecar(rows: list[dict[str, object]], schema: pa.Schema, target: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp = target.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)


def _require_columns(actual: Iterable[str], required: Iterable[str], source: Path | str) -> None:
    actual_set = set(actual)
    missing = [column for column in required if column not in actual_set]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {', '.join(missing)}")


def _require_fraction(value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError("min_covered_regulator_fraction must be between 0 and 1")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _require_text(value: object, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _text(value: object) -> str:
    return str(value or "").strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
