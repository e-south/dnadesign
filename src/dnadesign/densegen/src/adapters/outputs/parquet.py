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
from ...utils.logging_utils import install_native_stderr_filters
from .base import DEFAULT_NAMESPACE, AlignmentDigest, SinkBase
from .id_index import INDEX_FILENAME, IdIndex
from .record import OutputRecord

install_native_stderr_filters(suppress_solver_messages=False)


def _meta_arrow_type(name: str, pa):
    list_str = {
        "tf_list",
        "tfbs_parts",
        "used_tfbs",
        "used_tf_list",
        "input_pwm_ids",
        "required_regulators",
    }
    list_float = {
        "input_pwm_pvalue_bins",
    }
    list_int = {
        "input_pwm_mining_retain_bin_ids",
    }
    int_fields = {
        "length",
        "random_seed",
        "solver_threads",
        "min_count_per_tf",
        "min_required_regulators",
        "input_pwm_n_sites",
        "input_pwm_oversample_factor",
        "input_pwm_mining_batch_size",
        "input_pwm_mining_max_batches",
        "input_pwm_mining_max_candidates",
        "input_pwm_mining_log_every_batches",
        "input_row_count",
        "input_tf_count",
        "input_tfbs_count",
        "input_tf_tfbs_pair_count",
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
        "pad_bases",
        "pad_attempts",
    }
    float_fields = {
        "compression_ratio",
        "input_pwm_score_threshold",
        "input_pwm_score_percentile",
        "input_pwm_pvalue_threshold",
        "input_pwm_mining_max_seconds",
        "sampling_fraction",
        "sampling_fraction_pairs",
        "pad_gc_min",
        "pad_gc_max",
        "pad_gc_target_min",
        "pad_gc_target_max",
        "pad_gc_actual",
        "gc_total",
        "gc_core",
        "solver_objective",
        "solver_solve_time_s",
        "solver_time_limit_seconds",
    }
    bool_fields = {
        "covers_all_tfs_in_solution",
        "covers_required_regulators",
        "sampling_relaxed_cap",
        "pad_used",
        "pad_relaxed",
        "input_pwm_keep_all_candidates_debug",
        "input_pwm_include_matched_sequence",
    }

    if name in list_str:
        return pa.list_(pa.string())
    if name in list_float:
        return pa.list_(pa.float64())
    if name in list_int:
        return pa.list_(pa.int64())
    if name == "used_tfbs_detail":
        return pa.list_(
            pa.struct(
                [
                    pa.field("tf", pa.string()),
                    pa.field("tfbs", pa.string()),
                    pa.field("motif_id", pa.string()),
                    pa.field("tfbs_id", pa.string()),
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
    if not path.exists():
        return
    if path.is_dir():
        raise ValueError(
            "Parquet output path must be a file (single parquet). "
            f"Found directory at {path}. Update output.parquet.path to a file."
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover - depends on optional pyarrow install
        raise RuntimeError(f"Parquet support is not available: {e}") from e

    expected = _build_schema(namespace, pa)
    existing = pq.ParquetFile(path).schema_arrow
    missing, extra, mismatched = _schema_mismatch_details(expected, existing)
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        if mismatched:
            details.append(f"type_mismatch={mismatched[:3]}")
        hint = " ".join(details)
        raise RuntimeError(
            "Existing Parquet schema does not match the current DenseGen schema. "
            f"file={path} {hint}. Remove the output file or write to a new path. "
            "Legacy JSON-encoded metadata columns are not supported."
        )


def consolidate_parquet_parts(
    final_path: Path,
    *,
    part_glob: str,
    namespace: str = DEFAULT_NAMESPACE,
    batch_size: int = 4096,
) -> bool:
    parts = sorted(final_path.parent.glob(part_glob))
    if not parts:
        return False
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Parquet support is not available: {e}") from e

    sources = [str(p) for p in parts]
    if final_path.exists():
        sources.insert(0, str(final_path))
    dataset = ds.dataset(sources, format="parquet")
    schema = _build_schema(namespace, pa)
    tmp_path = final_path.with_suffix(".tmp")
    writer = pq.ParquetWriter(tmp_path, schema=schema)
    scanner = ds.Scanner.from_dataset(dataset, batch_size=batch_size)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        writer.write_table(pa.Table.from_batches([batch], schema=schema))
    writer.close()
    tmp_path.replace(final_path)
    for part in parts:
        part.unlink()
    return True


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
        index_path: str | Path | None = None,
    ):
        self.final_path = Path(path)
        if self.final_path.exists() and self.final_path.is_dir():
            raise ValueError(f"Parquet output path must be a file, got directory: {self.final_path}")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
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
        if index_path is None:
            index_path = self.final_path.parent / INDEX_FILENAME
        self._index = IdIndex(Path(index_path))
        self._part_glob = f"{self.final_path.stem}__part-*.parquet"

        if self.final_path.exists():
            validate_parquet_schema(self.final_path, namespace=self.namespace)
            self._index.bootstrap_from_parquet(self.final_path)
        for part in sorted(self.final_path.parent.glob(self._part_glob)):
            self._index.bootstrap_from_parquet(part)

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

        file_path = self.final_path.parent / f"{self.final_path.stem}__part-{uuid.uuid4().hex}.parquet"
        pq.write_table(table, file_path)
        self._index.add([row["id"] for row in self._buf])
        self._seen_ids.clear()
        self._buf.clear()

    def existing_ids(self) -> set[str]:
        return set(self._index.existing_ids()).union(self._seen_ids)

    def alignment_digest(self) -> AlignmentDigest | None:
        return self._index.alignment_digest()

    def finalize(self) -> None:
        self.flush()
        consolidate_parquet_parts(
            self.final_path,
            part_glob=self._part_glob,
            namespace=self.namespace,
        )
