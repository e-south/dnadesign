"""CLI mutation-gate tests for the SFXI reference-overlay recipe."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.sfxi_reference_overlay import cli
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.sfxi_reference_overlay.recipe import (
    OverlayPreview,
)


def test_cli_write_uses_explicit_publication_gate(monkeypatch, capsys, tmp_path: Path) -> None:
    preview = OverlayPreview(
        table=pa.table({"id": ["usr-a"]}),
        source_ref="reader-record-selection:fixture@sha256:digest",
        record_digests=("sha256:record",),
        dataset_name="dataset-a",
    )
    calls: list[tuple[Path, OverlayPreview]] = []
    monkeypatch.setattr(cli, "build_overlay_preview", lambda **_: preview)
    monkeypatch.setattr(
        cli,
        "publish_overlay",
        lambda *, usr_root, preview: calls.append((usr_root, preview)) or 1,
    )

    assert cli.main(["--reader-root", str(tmp_path / "reader"), "--expected-count", "1", "--write"]) == 0

    assert calls == [(cli.default_usr_root(), preview)]
    assert json.loads(capsys.readouterr().out)["written"] is True


def test_cli_defaults_to_read_only(monkeypatch, capsys, tmp_path: Path) -> None:
    preview = OverlayPreview(
        table=pa.table({"id": ["usr-a"]}),
        source_ref="reader-record-selection:fixture@sha256:digest",
        record_digests=("sha256:record",),
        dataset_name="dataset-a",
    )

    def fail_if_published(**_) -> int:
        raise AssertionError("unexpected write")

    monkeypatch.setattr(cli, "build_overlay_preview", lambda **_: preview)
    monkeypatch.setattr(cli, "publish_overlay", fail_if_published)

    assert cli.main(["--reader-root", str(tmp_path / "reader"), "--expected-count", "1"]) == 0

    assert json.loads(capsys.readouterr().out)["written"] is False
