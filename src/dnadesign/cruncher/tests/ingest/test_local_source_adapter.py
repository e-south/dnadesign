"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/ingest/test_local_source_adapter.py

Regression tests for local source adapter Cruncher ingest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dnadesign.cruncher.app.fetch_service import fetch_motifs
from dnadesign.cruncher.ingest.adapters.local import LocalMotifAdapter, LocalMotifAdapterConfig
from dnadesign.cruncher.ingest.models import MotifQuery, SiteQuery
from dnadesign.cruncher.store.catalog_index import CatalogIndex


def _write_meme(path: Path) -> None:
    path.write_text(
        "MEME version 4.12.0\n"
        "ALPHABET= ACGT\n"
        "MOTIF MEME-1 demo\n"
        "letter-probability matrix: alength= 4 w= 2 nsites= 5 E= 1e-3\n"
        "0.2 0.3 0.1 0.4\n"
        "0.25 0.25 0.25 0.25\n"
    )


def _make_adapter(root: Path, *, extra_modules: tuple[str, ...] = ()) -> LocalMotifAdapter:
    cfg = LocalMotifAdapterConfig(
        source_id="local",
        root=root,
        patterns=("*.txt",),
        recursive=False,
        format_map={".txt": "MEME"},
        default_format=None,
        tf_name_strategy="stem",
        matrix_semantics="probabilities",
        extra_parser_modules=extra_modules,
    )
    return LocalMotifAdapter(cfg)


def _write_meme_blocks(path: Path) -> None:
    path.write_text(
        "MEME version 4.12.0\n"
        "ALPHABET= ACGT\n"
        "MOTIF MEME-1 cusR\n"
        "letter-probability matrix: alength= 4 w= 3 nsites= 2 E= 1e-3\n"
        "0.25 0.25 0.25 0.25\n"
        "0.25 0.25 0.25 0.25\n"
        "0.25 0.25 0.25 0.25\n"
        "Motif 1 sites in BLOCKS format\n"
        "seq1 (10) ACG\n"
        "seq2 (20) ATG\n"
    )


def test_local_adapter_fetches_motifs(tmp_path: Path) -> None:
    root = tmp_path / "motifs"
    root.mkdir()
    _write_meme(root / "cusR.txt")
    _write_meme(root / "lexA.txt")

    adapter = _make_adapter(root)
    catalog_root = tmp_path / ".cruncher"
    written = fetch_motifs(adapter, catalog_root, names=["cusR"], motif_ids=None)

    assert written
    catalog = CatalogIndex.load(catalog_root)
    entry = catalog.entries.get("local:cusR")
    assert entry is not None
    assert entry.tf_name == "cusR"
    assert entry.has_matrix is True


def test_tf_name_preserves_case_from_stem(tmp_path: Path) -> None:
    root = tmp_path / "motifs"
    root.mkdir()
    _write_meme(root / "CusR.txt")

    adapter = _make_adapter(root)
    descriptors = adapter.list_motifs(MotifQuery())
    assert descriptors[0].tf_name == "CusR"
    record = adapter.get_motif("cusR")
    assert record.descriptor.tf_name == "CusR"


def test_local_adapter_no_matching_files(tmp_path: Path) -> None:
    root = tmp_path / "motifs"
    root.mkdir()
    cfg = LocalMotifAdapterConfig(
        source_id="local",
        root=root,
        patterns=("*.meme",),
        recursive=False,
        format_map={".meme": "MEME"},
        default_format=None,
        tf_name_strategy="stem",
        matrix_semantics="probabilities",
    )
    with pytest.raises(FileNotFoundError):
        LocalMotifAdapter(cfg)


def test_extra_parser_module_import(tmp_path: Path) -> None:
    module_path = tmp_path / "custom_parsers.py"
    module_path.write_text(
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from dnadesign.cruncher.core.pwm import PWM\n"
        "from dnadesign.cruncher.io.parsers.backend import register\n"
        "\n"
        "@register('CUSTOM')\n"
        "def parse_custom(path: Path) -> PWM:\n"
        "    _ = path.read_text()\n"
        "    return PWM(name='custom', matrix=np.array([[0.25,0.25,0.25,0.25]]))\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        root = tmp_path / "motifs"
        root.mkdir()
        file_path = root / "abc.custom"
        file_path.write_text("dummy")
        cfg = LocalMotifAdapterConfig(
            source_id="local",
            root=root,
            patterns=("*.custom",),
            recursive=False,
            format_map={".custom": "CUSTOM"},
            default_format=None,
            tf_name_strategy="stem",
            matrix_semantics="probabilities",
            extra_parser_modules=("custom_parsers",),
        )
        adapter = LocalMotifAdapter(cfg)
        record = adapter.get_motif("abc")
        assert record.descriptor.motif_id == "abc"
        assert np.allclose(record.matrix[0], [0.25, 0.25, 0.25, 0.25])
    finally:
        sys.path.remove(str(tmp_path))


def test_local_adapter_extracts_sites_from_meme_blocks(tmp_path: Path) -> None:
    root = tmp_path / "motifs"
    root.mkdir()
    _write_meme_blocks(root / "cusR.txt")

    cfg = LocalMotifAdapterConfig(
        source_id="local",
        root=root,
        patterns=("*.txt",),
        recursive=False,
        format_map={".txt": "MEME"},
        default_format=None,
        tf_name_strategy="stem",
        matrix_semantics="probabilities",
        extract_sites=True,
    )
    adapter = LocalMotifAdapter(cfg)
    sites = list(adapter.list_sites(SiteQuery(motif_id="cusR")))
    assert len(sites) == 2
    assert sites[0].sequence == "ACG"
    assert sites[0].evidence["sequence_name"] == "seq1"
    assert sites[0].evidence["start"] == 10
    assert sites[0].provenance.tags["record_kind"] == "meme_blocks"
    sites_for_motif = list(adapter.get_sites_for_motif("cusR", SiteQuery()))
    assert len(sites_for_motif) == 2


def test_local_adapter_ingests_ddg_table_via_extra_parser_module(tmp_path: Path) -> None:
    root = tmp_path / "motifs"
    root.mkdir()
    (root / "tetR.tsv").write_text(
        "PO\tA\tT\tC\tG\n"
        "1\t0.164072\t0.519334\t-0.0468541\t0.22966\n"
        "2\t0.642569\t0.164072\t0.188965\t0.943314\n"
        "3\t1.04481\t0.558693\t0.164072\t0.848264\n"
        "4\t0.959646\t0.164072\t2.17536\t0.718959\n"
        "5\t0.164072\t1.19894\t1.79469\t1.56685\n"
        "6\t1.46463\t0.164072\t1.74384\t1.54845\n"
        "7\t0.966397\t1.07557\t0.164072\t1.72883\n"
        "8\t0.164072\t0.504312\t0.755195\t0.148902\n"
        "9\t0.0876645\t0.164072\t0\t0.0156484\n"
        "10\t0.289722\t0.164072\t0.00707563\t0.843474\n"
        "11\t1.77568\t1.3447\t2.34804\t0.164072\n"
        "12\t0.164072\t1.72354\t1.62782\t1.49176\n"
        "13\t0.877518\t0.164072\t1.30879\t1.9052\n"
        "14\t0.164072\t0.544642\t0.3387\t1.7537\n"
        "15\t0.540091\t0.8821\t1.0861\t0.164072\n"
        "16\t-0.224147\t0.280358\t0.442769\t0.164072\n"
        "17\t0.31238\t-0.0651059\t0.164072\t-0.303844\n",
        encoding="utf-8",
    )

    cfg = LocalMotifAdapterConfig(
        source_id="westmann_tetr_mitomi",
        root=root,
        patterns=("*.tsv",),
        recursive=False,
        format_map={".tsv": "DDG_TABLE"},
        default_format=None,
        tf_name_strategy="stem",
        matrix_semantics="probabilities",
        extra_parser_modules=("dnadesign.cruncher.io.parsers.ddg_table",),
    )
    adapter = LocalMotifAdapter(cfg)
    catalog_root = tmp_path / ".cruncher"
    written = fetch_motifs(adapter, catalog_root, names=["tetR"], motif_ids=None)

    assert written
    catalog = CatalogIndex.load(catalog_root)
    entry = catalog.entries.get("westmann_tetr_mitomi:tetR")
    assert entry is not None
    assert entry.tf_name == "tetR"
    assert entry.matrix_length == 17
    assert entry.has_matrix is True
