"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_regulondb_ssl.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapterConfig, _build_ssl_context


def _common_names(certs: list[dict]) -> set[str]:
    names: set[str] = set()
    for cert in certs:
        for entry in cert.get("subject", ()):
            for key, value in entry:
                if key == "commonName":
                    names.add(value)
    return names


def test_regulondb_ssl_context_includes_intermediate() -> None:
    context = _build_ssl_context(RegulonDBAdapterConfig())
    assert "GlobalSign RSA OV SSL CA 2018" in _common_names(context.get_ca_certs())
