"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_sequence_ingress.py

Raw, text, and FASTA ingress tests for canonical Junction requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dnadesign.junction.cli as cli_module
import dnadesign.junction.ingress as ingress_module
import dnadesign.junction.publication.fasta as fasta_module
from dnadesign.junction import load_sequence_records, request_from_sequences, sequence_record
from dnadesign.junction.cli import app
from dnadesign.junction.contracts.request import parse_request
from dnadesign.junction.errors import JunctionConfigError
from dnadesign.junction.tests.test_planner import _request_mapping

runner = CliRunner()


def test_sequence_ingress_and_export_modules_remain_bounded() -> None:
    ingress_root = Path(ingress_module.__file__).parent
    budgets = {
        Path(cli_module.__file__): 230,
        ingress_root / "request.py": 140,
        ingress_root / "sources.py": 180,
        Path(fasta_module.__file__): 100,
    }

    for path, budget in budgets.items():
        assert len(path.read_text(encoding="utf-8").splitlines()) <= budget, f"{path.name} exceeds {budget} lines"


def _base_request_file(tmp_path: Path) -> Path:
    path = tmp_path / "base-request.json"
    path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    return path


def _read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    identifier: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            identifier = line[1:].split(maxsplit=1)[0]
            records[identifier] = []
        elif identifier is not None:
            records[identifier].append(line)
    return {key: "".join(lines) for key, lines in records.items()}


def test_raw_sequence_normalizes_case_and_whitespace() -> None:
    record = sequence_record(" acgt\nACGT ", target_id="target-a")

    assert record.id == "target-a"
    assert record.sequence == "ACGTACGT"


def test_fasta_ingress_preserves_record_ids_and_normalizes_sequences(tmp_path: Path) -> None:
    source = tmp_path / "targets.fasta"
    source.write_text(">target-a first target\nacgt\nACGT\n>target-b\nTGCATGCA\n", encoding="utf-8")

    records = load_sequence_records(source)

    assert [(record.id, record.sequence) for record in records] == [
        ("target-a", "ACGTACGT"),
        ("target-b", "TGCATGCA"),
    ]


def test_text_ingress_accepts_one_wrapped_sequence(tmp_path: Path) -> None:
    source = tmp_path / "target.txt"
    source.write_text("acgt acgt\nacgt\n", encoding="utf-8")

    [record] = load_sequence_records(source, target_id="target-a")

    assert record.id == "target-a"
    assert record.sequence == "ACGTACGTACGT"


def test_fasta_ingress_rejects_duplicate_ids_and_ambiguity_codes(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.fasta"
    duplicate.write_text(">target-a\nACGT\n>target-a\nTGCA\n", encoding="utf-8")
    ambiguous = tmp_path / "ambiguous.fasta"
    ambiguous.write_text(">target-a\nACNT\n", encoding="utf-8")

    with pytest.raises(JunctionConfigError, match="duplicate record identifiers"):
        load_sequence_records(duplicate)
    with pytest.raises(JunctionConfigError, match="uppercase ACGT"):
        load_sequence_records(ambiguous)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is unavailable")
def test_sequence_ingress_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("ACGT", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(JunctionConfigError, match="Unable to open Junction sequence input"):
        load_sequence_records(link)


def test_request_from_sequences_uses_explicit_policy_and_terminal_binding_length() -> None:
    base = parse_request(_request_mapping())
    source = base.targets[0]
    records = (sequence_record(source.sequence, target_id="target-a"),)

    request = request_from_sequences(
        records,
        planning=base.planning,
        order_policy=base.order_policy,
        seed=base.seed,
        primer_binding_length=20,
        assembly_group_id="assembly-a",
    )

    [target] = request.targets
    assert target.id == "target-a"
    assert target.assembly_group_id == "assembly-a"
    assert target.recovery_primers.forward.binding_sequence == target.sequence[:20]
    assert len(target.recovery_primers.reverse.binding_sequence) == 20


def test_request_from_sequences_rejects_false_universal_primers() -> None:
    base = parse_request(_request_mapping())
    source = base.targets[0].sequence
    records = (
        sequence_record(source, target_id="target-a"),
        sequence_record("T" + source[1:], target_id="target-b"),
    )

    with pytest.raises(JunctionConfigError, match="universal recovery requires"):
        request_from_sequences(
            records,
            planning=base.planning,
            order_policy=base.order_policy,
            seed=base.seed,
            primer_binding_length=20,
            recovery_mode="universal",
        )


def test_cli_request_compiles_raw_sequence_to_canonical_json(tmp_path: Path) -> None:
    base = parse_request(_request_mapping())
    result = runner.invoke(
        app,
        [
            "request",
            "--base-request",
            str(_base_request_file(tmp_path)),
            "--sequence",
            base.targets[0].sequence.lower(),
            "--target-id",
            "target-raw",
            "--primer-binding-length",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    request = parse_request(json.loads(result.stdout))
    assert request.targets[0].id == "target-raw"
    assert request.targets[0].sequence == base.targets[0].sequence


def test_cli_request_compiles_fasta_and_rejects_ambiguous_source_choice(tmp_path: Path) -> None:
    base = parse_request(_request_mapping())
    source = tmp_path / "targets.fasta"
    source.write_text(f">target-a\n{base.targets[0].sequence}\n", encoding="utf-8")
    base_request = _base_request_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "request",
            "--base-request",
            str(base_request),
            "--input",
            str(source),
            "--primer-binding-length",
            "20",
        ],
    )
    ambiguous = runner.invoke(
        app,
        [
            "request",
            "--base-request",
            str(base_request),
            "--input",
            str(source),
            "--sequence",
            base.targets[0].sequence,
            "--primer-binding-length",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert parse_request(json.loads(result.stdout)).targets[0].id == "target-a"
    assert ambiguous.exit_code == 1
    assert "Provide exactly one" in ambiguous.stderr


def test_cli_fasta_to_verified_bundle_dogfood(tmp_path: Path) -> None:
    base = parse_request(_request_mapping())
    source = tmp_path / "targets.fasta"
    source.write_text(f">target-a\n{base.targets[0].sequence}\n", encoding="utf-8")
    compiled = runner.invoke(
        app,
        [
            "request",
            "--base-request",
            str(_base_request_file(tmp_path)),
            "--input",
            str(source),
            "--primer-binding-length",
            "20",
        ],
    )
    assert compiled.exit_code == 0, compiled.output
    request_path = tmp_path / "request.json"
    request_path.write_text(compiled.stdout, encoding="utf-8")
    bundle = tmp_path / "bundle"

    built = runner.invoke(app, ["build", str(request_path), "--output", str(bundle), "--format", "json"])
    verified = runner.invoke(app, ["verify", str(bundle), "--format", "json"])

    assert built.exit_code == 0, built.output
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["artifact_count"] == 9
    assert (bundle / "orders" / "oligos.tsv").is_file()
    assert (bundle / "sequences" / "targets.fasta").is_file()
    assert (bundle / "sequences" / "oligos.fasta").is_file()
    assert (bundle / "sequences" / "expected_pcr_products.fasta").is_file()
    request = parse_request(json.loads(compiled.stdout))
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    assert _read_fasta(bundle / "sequences" / "targets.fasta") == {
        target.id: target.sequence for target in request.targets
    }
    assert _read_fasta(bundle / "sequences" / "oligos.fasta") == {
        order["order_id"]: order["sequence"] for order in plan["orders"]
    }
    assert _read_fasta(bundle / "sequences" / "expected_pcr_products.fasta") == {
        target["target_id"]: target["recovery"]["extended_top_strand"] for target in plan["targets"]
    }
