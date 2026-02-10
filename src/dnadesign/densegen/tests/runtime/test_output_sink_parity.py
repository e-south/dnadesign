"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_output_sink_parity.py

Parity checks for DenseGen metadata between Parquet and USR sinks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.adapters.outputs.parquet import ParquetSink
from dnadesign.densegen.src.adapters.outputs.record import OutputRecord
from dnadesign.densegen.src.adapters.outputs.usr_writer import USRWriter
from dnadesign.densegen.src.core.metadata_schema import META_FIELDS
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _make_usr_writer(tmp_path: Path, *, deduplicate: bool = True) -> tuple[Path, USRWriter]:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        allow_overwrite=False,
        deduplicate=deduplicate,
    )
    return root, writer


def test_parquet_and_usr_sinks_keep_same_metadata_and_stage_a_lineage(tmp_path: Path) -> None:
    root, usr_writer = _make_usr_writer(tmp_path)
    parquet_path = tmp_path / "dense_arrays.parquet"
    parquet_sink = ParquetSink(path=str(parquet_path), namespace="densegen", chunk_size=1)
    meta = output_meta(library_hash="demo_hash", library_index=1)
    meta["used_tfbs_detail"] = [
        {
            "tf": "lexA",
            "tfbs": "AAA",
            "orientation": "fwd",
            "offset": 0,
            "stage_a_best_hit_score": 8.25,
            "stage_a_rank_within_regulator": 2,
            "stage_a_tier": 1,
            "stage_a_fimo_start": 14,
            "stage_a_fimo_stop": 17,
            "stage_a_fimo_strand": "+",
            "stage_a_selection_rank": 3,
            "stage_a_selection_score_norm": 0.87,
            "stage_a_tfbs_core": "AAA",
        }
    ]
    rec = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert parquet_sink.add(rec) is True
    assert usr_writer.add(rec) is True
    parquet_sink.finalize()
    usr_writer.finalize()

    parquet_table = pq.read_table(parquet_path)
    usr_parts = sorted((root / "demo" / "_derived" / "densegen").glob("part-*.parquet"))
    assert usr_parts
    usr_table = pq.read_table(usr_parts[-1])

    shared_columns = [f"densegen__{field.name}" for field in META_FIELDS]
    for column in shared_columns:
        assert column in parquet_table.column_names
        assert column in usr_table.column_names
        assert parquet_table.column(column).to_pylist() == usr_table.column(column).to_pylist()

    parquet_detail = parquet_table.column("densegen__used_tfbs_detail").to_pylist()[0][0]
    usr_detail = usr_table.column("densegen__used_tfbs_detail").to_pylist()[0][0]
    for detail in (parquet_detail, usr_detail):
        assert detail["stage_a_best_hit_score"] == 8.25
        assert detail["stage_a_rank_within_regulator"] == 2
        assert detail["stage_a_tier"] == 1
        assert detail["stage_a_fimo_start"] == 14
        assert detail["stage_a_fimo_stop"] == 17
        assert detail["stage_a_fimo_strand"] == "+"
        assert detail["stage_a_selection_rank"] == 3
        assert detail["stage_a_selection_score_norm"] == 0.87
        assert detail["stage_a_tfbs_core"] == "AAA"


def test_usr_writer_dedupes_without_full_parquet_id_preload(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    ds = usr_pkg.Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {
                "sequence": "ACGTACGTAA",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "seed",
            }
        ],
        strict_id_check=True,
    )
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        allow_overwrite=False,
    )
    rec = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo_hash", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("pyarrow.parquet.read_table should not be used for USRWriter dedupe")

    monkeypatch.setattr("pyarrow.parquet.read_table", _boom)
    assert writer.add(rec) is False


def test_usr_writer_passes_explicit_actor_to_usr_mutations(tmp_path: Path, monkeypatch) -> None:
    _root, writer = _make_usr_writer(tmp_path)
    meta = output_meta(library_hash="demo_hash", library_index=1)
    meta["run_id"] = "run-actor"
    rec = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    calls: list[tuple[str, dict | None]] = []

    def _capture_import(rows, **kwargs):
        calls.append(("import_rows", kwargs.get("actor")))
        return len(rows)

    def _capture_overlay(namespace, table, **kwargs):
        _ = namespace
        rows = int(getattr(table, "num_rows", 0))
        calls.append(("write_overlay_part", kwargs.get("actor")))
        return rows

    def _capture_event(action, **kwargs):
        calls.append((f"log:{action}", kwargs.get("actor")))
        return None

    monkeypatch.setattr(writer.ds, "import_rows", _capture_import)
    monkeypatch.setattr(writer.ds, "write_overlay_part", _capture_overlay)
    monkeypatch.setattr(writer.ds, "log_event", _capture_event)

    assert writer.add(rec) is True
    writer.finalize()

    expected_run = "run-actor"
    for action in {"import_rows", "write_overlay_part", "log:densegen_health"}:
        actor = next((actor for name, actor in calls if name == action), None)
        assert isinstance(actor, dict)
        assert actor.get("tool") == "densegen"
        assert actor.get("run_id") == expected_run


def test_usr_writer_updates_id_index_when_dedup_disabled(tmp_path: Path) -> None:
    _root, writer = _make_usr_writer(tmp_path, deduplicate=False)
    rec = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo_hash", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(rec) is True
    writer.finalize()
    assert rec.id in writer.existing_ids()
    digest = writer.alignment_digest()
    assert digest.id_count >= 1


def test_usr_writer_flush_uses_batch_lookup_for_dedup(tmp_path: Path, monkeypatch) -> None:
    _root, writer = _make_usr_writer(tmp_path, deduplicate=True)
    recs = [
        OutputRecord.from_sequence(
            sequence="ACGTACGTAA",
            meta=output_meta(library_hash="demo_hash", library_index=1),
            source="unit",
            bio_type="dna",
            alphabet="dna_4",
        ),
        OutputRecord.from_sequence(
            sequence="TGCATGCATT",
            meta=output_meta(library_hash="demo_hash", library_index=1),
            source="unit",
            bio_type="dna",
            alphabet="dna_4",
        ),
    ]
    writer._records = list(recs)
    writer._seen_ids = {rec.id for rec in recs}
    monkeypatch.setattr(
        writer._id_index,
        "contains",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("flush should use batch lookup")),
    )
    monkeypatch.setattr(writer._id_index, "contains_many", lambda ids: set(ids), raising=False)
    writer.flush()
