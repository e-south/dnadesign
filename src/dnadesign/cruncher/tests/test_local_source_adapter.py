"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_local_source_adapter.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dnadesign.cruncher.ingest.adapters.local import LocalMotifAdapter, LocalMotifAdapterConfig
from dnadesign.cruncher.ingest.models import MotifQuery, SiteQuery
from dnadesign.cruncher.services.fetch_service import fetch_motifs
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
