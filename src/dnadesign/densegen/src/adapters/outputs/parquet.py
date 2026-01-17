"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/outputs/parquet.py

Parquet output sink for DenseGen (dataset directory).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ...core.metadata_schema import META_FIELDS, validate_metadata
from .base import DEFAULT_NAMESPACE, AlignmentDigest, SinkBase
from .id_index import IdIndex
from .record import OutputRecord


def _meta_arrow_type(name: str, pa):
    list_str = {
        "solver_options",
        "tf_list",
        "tfbs_parts",
        "used_tfbs",
        "used_tf_list",
        "input_pwm_ids",
        "required_regulators",
    }
    int_fields = {
        "length",
        "random_seed",
        "min_count_per_tf",
        "min_required_regulators",
        "input_pwm_n_sites",
        "input_pwm_oversample_factor",
        "input_row_count",
        "input_tf_count",
        "input_tfbs_count",
        "library_size",
        "library_unique_tf_count",
        "library_unique_tfbs_count",
        "sequence_length",
        "sampling_target_length",
        "sampling_achieved_length",
        "sampling_final_cap",
        "sampling_library_size",
        "sampling_iterative_max_libraries",
        "sampling_iterative_min_new_solutions",
        "sampling_library_index",
        "gap_fill_bases",
        "gap_fill_attempts",
    }
    float_fields = {
        "compression_ratio",
        "input_pwm_score_threshold",
        "input_pwm_score_percentile",
        "sampling_fraction",
        "gap_fill_gc_min",
        "gap_fill_gc_max",
        "gap_fill_gc_target_min",
        "gap_fill_gc_target_max",
        "gap_fill_gc_actual",
        "gc_total",
        "gc_core",
        "solver_objective",
        "solver_solve_time_s",
    }
    bool_fields = {
        "covers_all_tfs_in_solution",
        "covers_required_regulators",
        "sampling_relaxed_cap",
        "gap_fill_used",
        "gap_fill_relaxed",
    }

    if name in list_str:
        return pa.list_(pa.string())
    if name == "used_tfbs_detail":
        return pa.list_(
            pa.struct(
                [
                    pa.field("tf", pa.string()),
                    pa.field("tfbs", pa.string()),
                    pa.field("orientation", pa.string()),
                    pa.field("offset", pa.int64()),
                    pa.field("offset_raw", pa.int64()),
                    pa.field("length", pa.int64()),
                    pa.field("end", pa.int64()),
                    pa.field("pad_left", pa.int64()),
                    pa.field("site_id", pa.string()),
                    pa.field("source", pa.string()),
                ]
            )
        )
    if name == "used_tf_counts":
        return pa.list_(pa.struct([pa.field("tf", pa.string()), pa.field("count", pa.int64())]))
    if name == "min_count_by_regulator":
        return pa.list_(
            pa.struct(
                [
                    pa.field("tf", pa.string()),
                    pa.field("min_count", pa.int64()),
                ]
            )
        )
    if name == "fixed_elements":
        promoter = pa.list_(
            pa.struct(
                [
                    pa.field("name", pa.string()),
                    pa.field("upstream", pa.string()),
                    pa.field("downstream", pa.string()),
                    pa.field("spacer_length", pa.list_(pa.int64())),
                    pa.field("upstream_pos", pa.list_(pa.int64())),
                    pa.field("downstream_pos", pa.list_(pa.int64())),
                ]
            )
        )
        side_biases = pa.struct(
            [
                pa.field("left", pa.list_(pa.string())),
                pa.field("right", pa.list_(pa.string())),
            ]
        )
        return pa.struct([pa.field("promoter_constraints", promoter), pa.field("side_biases", side_biases)])
    if name in int_fields:
        return pa.int64()
    if name in float_fields:
        return pa.float64()
    if name in bool_fields:
        return pa.bool_()
    return pa.string()


def _build_schema(namespace: str, pa):
    fields = [
        pa.field("id", pa.string()),
        pa.field("sequence", pa.string()),
        pa.field("bio_type", pa.string()),
        pa.field("alphabet", pa.string()),
        pa.field("source", pa.string()),
    ]
    for field in META_FIELDS:
        meta_name = f"{namespace}__{field.name}"
        fields.append(pa.field(meta_name, _meta_arrow_type(field.name, pa)))
    return pa.schema(fields)


def _schema_mismatch_details(expected, existing) -> tuple[list[str], list[str], list[str]]:
    exp_map = {field.name: field.type for field in expected}
    got_map = {field.name: field.type for field in existing}
    missing = sorted(set(exp_map) - set(got_map))
    extra = sorted(set(got_map) - set(exp_map))
    mismatched = []
    for name in sorted(set(exp_map) & set(got_map)):
        if exp_map[name] != got_map[name]:
            mismatched.append(f"{name} (expected {exp_map[name]}, got {got_map[name]})")
    return missing, extra, mismatched


def validate_parquet_schema(path: Path, *, namespace: str = DEFAULT_NAMESPACE) -> None:
    if not any(path.glob("*.parquet")):
        return
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception as e:  # pragma: no cover - depends on optional pyarrow install
        raise RuntimeError(f"Parquet support is not available: {e}") from e

    dataset = ds.dataset(path, format="parquet")
    expected = _build_schema(namespace, pa)
    for frag in dataset.get_fragments():
        frag_schema = getattr(frag, "physical_schema", None) or dataset.schema
        missing, extra, mismatched = _schema_mismatch_details(expected, frag_schema)
        if missing or extra or mismatched:
            details = []
            if missing:
                details.append(f"missing={missing[:5]}")
            if extra:
                details.append(f"extra={extra[:5]}")
            if mismatched:
                details.append(f"type_mismatch={mismatched[:3]}")
            hint = " ".join(details)
            path_label = getattr(frag, "path", "<unknown>")
            raise RuntimeError(
                "Existing Parquet dataset schema does not match the current DenseGen schema. "
                f"file={path_label} {hint}. Remove the dataset or write to a new path. "
                "Legacy JSON-encoded metadata columns are not supported."
            )


class ParquetSink(SinkBase):
    def __init__(
        self,
        path: str,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        bio_type: str = "dna",
        alphabet: str = "dna_4",
        deduplicate: bool = True,
        chunk_size: int = 2048,
    ):
        self.path = Path(path)
        if self.path.exists() and self.path.is_file():
            raise ValueError(f"Parquet output path must be a directory, got file: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.bio_type = bio_type
        self.alphabet = alphabet
        self.deduplicate = deduplicate
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self._seen_ids: set[str] = set()
        self._schema = None
        self._buf: list[dict[str, Any]] = []
        self._index = IdIndex(self.path)

        validate_parquet_schema(self.path, namespace=self.namespace)
        if any(self.path.glob("*.parquet")):
            self._index.bootstrap_from_parquet(self.path)

    def add(self, record: OutputRecord) -> bool:
        if record.bio_type != self.bio_type or record.alphabet != self.alphabet:
            raise ValueError(
                "OutputRecord bio_type/alphabet mismatch for Parquet sink. "
                f"record=({record.bio_type}, {record.alphabet}) "
                f"sink=({self.bio_type}, {self.alphabet})"
            )
        validate_metadata(record.meta)
        if self.deduplicate and self._index.contains(record.id):
            return False
        if record.id in self._seen_ids:
            return False
        self._seen_ids.add(record.id)

        row = record.to_row(self.namespace)
        self._buf.append(row)

        if len(self._buf) >= self.chunk_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._buf:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:  # pragma: no cover - depends on optional pyarrow install
            raise RuntimeError(f"Parquet support is not available: {e}") from e

        # Ensure consistent keys across rows
        keys = set(self._buf[0].keys())
        for row in self._buf[1:]:
            if set(row.keys()) != keys:
                raise RuntimeError("Parquet output rows have inconsistent keys; check metadata schema.")

        if self._schema is None:
            self._schema = _build_schema(self.namespace, pa)
        table = pa.Table.from_pylist(self._buf, schema=self._schema)

        file_path = self.path / f"part-{uuid.uuid4().hex}.parquet"
        pq.write_table(table, file_path)
        self._index.add([row["id"] for row in self._buf])
        self._seen_ids.clear()
        self._buf.clear()

    def existing_ids(self) -> set[str]:
        return set(self._index.existing_ids()).union(self._seen_ids)

    def alignment_digest(self) -> AlignmentDigest | None:
        return self._index.alignment_digest()
