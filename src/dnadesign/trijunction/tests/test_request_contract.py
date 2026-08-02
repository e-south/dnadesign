from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

import dnadesign.trijunction as trijunction
from dnadesign.trijunction.contracts import TriJunctionConfigError
from dnadesign.trijunction.contracts import request as request_contract
from dnadesign.trijunction.contracts.request import (
    MAX_BARCODE_GENERATION_ATTEMPTS,
    MAX_BARCODE_SUBSET_ITERATIONS,
    MAX_MATCHING_ITERATIONS,
    MAX_REQUEST_BYTES,
    MAX_TOEHOLD_SEARCH_ITERATIONS,
    ComplementEndPreparation,
    Primer,
    RecoveryPrimerPair,
    load_request,
    parse_request,
    request_to_mapping,
)
from dnadesign.trijunction.contracts.request import files as request_files


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _request_mapping() -> dict[str, object]:
    sequence = "ACGTTGCA" * 10
    return {
        "schema": "dnadesign.trijunction.request.v1",
        "seed": 17,
        "planning": {
            "oligo_length": 60,
            "barcode_length": 12,
            "toehold_length": 10,
            "search_range": 20,
            "toehold_search_iterations": 50,
            "barcode_pool_factor": 5,
            "barcode_generation_attempts": 1000,
            "barcode_toehold_k": 5,
            "barcode_pair_k": 6,
            "barcode_subset_iterations": 100,
            "matching_iterations": 200,
            "barcode_gc_min": 0.35,
            "barcode_gc_max": 0.65,
            "barcode_max_homopolymer": 3,
        },
        "targets": [
            {
                "id": "target-b",
                "pool_id": "pool-2",
                "sequence": sequence,
                "recovery_primers": {
                    "mode": "target_specific",
                    "forward": {
                        "binding_sequence": sequence[:8],
                        "five_prime_extension": "GGCC",
                    },
                    "reverse": {
                        "binding_sequence": _reverse_complement(sequence[-8:]),
                        "five_prime_extension": "AATT",
                    },
                },
            },
            {
                "id": "target-a",
                "pool_id": "pool-1",
                "sequence": "TGCATGCA" * 10,
                "recovery_primers": {
                    "mode": "target_specific",
                    "forward": {
                        "binding_sequence": "TGCATGCA",
                        "five_prime_extension": "",
                    },
                    "reverse": {
                        "binding_sequence": "TGCATGCA",
                        "five_prime_extension": "",
                    },
                },
            },
        ],
        "order_policy": {
            "synthesis_scale": "25 nmol",
            "barcode_bearing_purification": "STD",
            "complement_purification": "PAGE",
            "primer_purification": "HPLC",
            "complement_end_preparation": "vendor_5_prime_phosphate",
            "max_oligo_length": 200,
        },
    }


def test_request_contract_exports_are_anchored_at_the_canonical_package() -> None:
    assert request_contract.load_request is load_request
    assert request_contract.parse_request is parse_request
    assert request_contract.request_to_mapping is request_to_mapping
    assert request_contract.Primer is Primer
    assert request_contract.RecoveryPrimerPair is RecoveryPrimerPair
    assert request_contract.ComplementEndPreparation is ComplementEndPreparation
    assert trijunction.Primer is Primer
    assert trijunction.RecoveryPrimerPair is RecoveryPrimerPair
    assert trijunction.ComplementEndPreparation is ComplementEndPreparation

    for removed_name in ("CodingEndPreparation", "RecoveryPrimers", "TargetSpec"):
        assert removed_name not in request_contract.__all__
        assert removed_name not in trijunction.__all__
        with pytest.raises(AttributeError):
            getattr(trijunction, removed_name)


def test_parse_request_is_immutable_and_canonicalizes_target_order() -> None:
    raw = _request_mapping()
    reversed_raw = {**raw, "targets": list(reversed(raw["targets"]))}  # type: ignore[arg-type]

    request = parse_request(raw)
    reordered = parse_request(reversed_raw)

    assert request == reordered
    assert [target.id for target in request.targets] == ["target-a", "target-b"]
    assert request_to_mapping(request) == request_to_mapping(reordered)
    assert request.to_mapping() == request_to_mapping(request)
    target_b = next(target for target in request.targets if target.id == "target-b")
    assert target_b.recovery_primers.forward.order_sequence == f"GGCC{target_b.sequence[:8]}"
    assert target_b.recovery_primers.reverse.order_sequence == (f"AATT{_reverse_complement(target_b.sequence[-8:])}")
    with pytest.raises(AttributeError):
        request.seed = 3  # type: ignore[misc]


def test_parse_request_accepts_the_universal_recovery_mode() -> None:
    raw = _request_mapping()
    raw["targets"] = [raw["targets"][0]]  # type: ignore[index]
    raw["targets"][0]["recovery_primers"]["mode"] = "universal"  # type: ignore[index]

    request = parse_request(raw)

    assert request.targets[0].recovery_primers.mode == "universal"


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_load_request_supports_yaml_and_json(tmp_path: Path, suffix: str) -> None:
    request_path = tmp_path / f"request{suffix}"
    raw = _request_mapping()
    if suffix == ".json":
        request_path.write_text(json.dumps(raw), encoding="utf-8")
    else:
        request_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_request(request_path)

    assert loaded.schema == "dnadesign.trijunction.request.v1"
    assert request_to_mapping(loaded)["targets"][0]["id"] == "target-a"  # type: ignore[index]
    assert sorted(tmp_path.iterdir()) == [request_path]


def test_load_request_opens_and_reads_one_bounded_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    real_open = os.open
    real_read = os.read
    open_calls: list[tuple[object, int]] = []
    read_limits: list[int] = []

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        open_calls.append((path, flags))
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    def recording_read(fd: int, size: int) -> bytes:
        read_limits.append(size)
        return real_read(fd, size)

    monkeypatch.setattr(request_files.os, "open", recording_open)
    monkeypatch.setattr(request_files.os, "read", recording_read)

    load_request(request_path)

    assert len(open_calls) == 1
    assert read_limits and max(read_limits) <= request_files._MAX_REQUEST_BYTES
    if hasattr(os, "O_NOFOLLOW"):
        assert open_calls[0][1] & os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        assert open_calls[0][1] & os.O_NONBLOCK


def test_load_request_rejects_descriptor_growth_after_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    real_fstat = os.fstat
    calls = 0

    def growing_fstat(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        observed = real_fstat(fd)
        if calls == 1:
            return observed
        fields = list(observed)
        fields[6] = request_files._MAX_REQUEST_BYTES + 1
        return os.stat_result(fields)

    monkeypatch.setattr(request_files.os, "fstat", growing_fstat)

    with pytest.raises(TriJunctionConfigError, match="exceeds.*input limit"):
        load_request(request_path)


def test_load_request_rejects_invalid_utf8_as_config_error(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_bytes(b"\xff")

    with pytest.raises(TriJunctionConfigError, match="UTF-8"):
        load_request(request_path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is unavailable")
def test_load_request_rejects_symlink(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(_request_mapping()), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.symlink_to(target_path)

    with pytest.raises(TriJunctionConfigError, match="Unable to open"):
        load_request(request_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.update(extra=True), "unknown field.*extra"),
        (lambda raw: raw.pop("seed"), "missing field.*seed"),
        (lambda raw: raw.update(schema="old"), "schema"),
        (lambda raw: raw.update(seed=-1), "seed"),
        (lambda raw: raw.update(seed=True), "seed"),
        (
            lambda raw: raw["planning"].update(extra=True),  # type: ignore[union-attr]
            "planning.*unknown field.*extra",
        ),
        (
            lambda raw: raw["planning"].update(oligo_length=53),  # type: ignore[union-attr]
            "oligo_length.*2.*barcode_length.*toehold_length.*search_range",
        ),
        (
            lambda raw: raw["planning"].update(barcode_gc_min=0.8, barcode_gc_max=0.2),  # type: ignore[union-attr]
            "barcode_gc_min.*barcode_gc_max",
        ),
        (lambda raw: raw.update(targets=[]), "at least one target"),
        (
            lambda raw: raw["targets"][0].update(sequence="acgt"),  # type: ignore[index,union-attr]
            "uppercase ACGT",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"].update(mode="mixed"),  # type: ignore[index,union-attr]
            "recovery_primers.mode",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"].update(mode="construct_specific"),  # type: ignore[index,union-attr]
            "target_specific.*universal",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"]["forward"].update(  # type: ignore[index,union-attr]
                binding_sequence="AAAAAAAA"
            ),
            "forward.*prefix",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"]["reverse"].update(  # type: ignore[index,union-attr]
                binding_sequence="AAAAAAAA"
            ),
            "reverse.*suffix",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"]["forward"].update(  # type: ignore[index,union-attr]
                five_prime_extension="ggcc"
            ),
            "five_prime_extension.*uppercase ACGT",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"]["forward"].pop(  # type: ignore[index,union-attr]
                "five_prime_extension"
            ),
            "forward.*missing field.*five_prime_extension",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"]["forward"].update(  # type: ignore[index,union-attr]
                annotation="downstream-only"
            ),
            "forward.*unknown field.*annotation",
        ),
        (
            lambda raw: raw["targets"][0]["recovery_primers"].update(  # type: ignore[index,union-attr]
                forward="ACGTTGCA"
            ),
            "forward.*object",
        ),
        (
            lambda raw: raw["targets"].append({**raw["targets"][0], "id": "target-c"}),  # type: ignore[index,union-attr]
            "duplicate sequence.*pool-2",
        ),
        (
            lambda raw: raw["targets"].append({**raw["targets"][0]}),  # type: ignore[index,union-attr]
            "duplicate target id",
        ),
        (
            lambda raw: raw["planning"].update(barcode_pool_factor=4),  # type: ignore[union-attr]
            "barcode_pool_factor.*at least 5",
        ),
        (
            lambda raw: raw["planning"].update(  # type: ignore[union-attr]
                toehold_search_iterations=MAX_TOEHOLD_SEARCH_ITERATIONS + 1
            ),
            "toehold_search_iterations.*100000.*request.v1",
        ),
        (
            lambda raw: raw["planning"].update(  # type: ignore[union-attr]
                barcode_generation_attempts=MAX_BARCODE_GENERATION_ATTEMPTS + 1
            ),
            "barcode_generation_attempts.*10000000.*request.v1",
        ),
        (
            lambda raw: raw["planning"].update(  # type: ignore[union-attr]
                barcode_subset_iterations=MAX_BARCODE_SUBSET_ITERATIONS + 1
            ),
            "barcode_subset_iterations.*100000.*request.v1",
        ),
        (
            lambda raw: raw["planning"].update(  # type: ignore[union-attr]
                matching_iterations=MAX_MATCHING_ITERATIONS + 1
            ),
            "matching_iterations.*100000.*request.v1",
        ),
        (
            lambda raw: raw["planning"].update(barcode_toehold_k=13),  # type: ignore[union-attr]
            "barcode_toehold_k.*barcode_length",
        ),
        (
            lambda raw: raw["planning"].update(barcode_pair_k=5),  # type: ignore[union-attr]
            "barcode_pair_k.*greater than barcode_toehold_k",
        ),
        (
            lambda raw: raw["order_policy"].update(complement_end_preparation="unknown"),  # type: ignore[union-attr]
            "complement_end_preparation",
        ),
        (
            lambda raw: raw["order_policy"].update(max_oligo_length=50),  # type: ignore[union-attr]
            "max_oligo_length.*oligo_length",
        ),
    ],
)
def test_parse_request_rejects_invalid_input(mutation: object, match: str) -> None:
    raw = _request_mapping()
    mutation(raw)  # type: ignore[operator]

    with pytest.raises(TriJunctionConfigError, match=match):
        parse_request(raw)


def test_parse_request_rejects_canonical_payload_above_file_limit() -> None:
    raw = _request_mapping()
    raw["order_policy"]["synthesis_scale"] = "x" * MAX_REQUEST_BYTES  # type: ignore[index]

    with pytest.raises(TriJunctionConfigError, match="canonical request exceeds.*input limit"):
        parse_request(raw)


@pytest.mark.parametrize(
    "legacy_field",
    ["barcode_purification", "coding_purification", "coding_end_preparation"],
)
def test_parse_request_rejects_legacy_order_policy_fields(legacy_field: str) -> None:
    raw = _request_mapping()
    raw["order_policy"][legacy_field] = "legacy"  # type: ignore[index]

    with pytest.raises(TriJunctionConfigError, match=rf"unknown field.*{legacy_field}"):
        parse_request(raw)


def test_parse_request_rejects_extension_bearing_primer_above_order_ceiling() -> None:
    raw = _request_mapping()
    target = raw["targets"][0]
    assert isinstance(target, dict)
    primers = target["recovery_primers"]
    assert isinstance(primers, dict)
    forward = primers["forward"]
    assert isinstance(forward, dict)
    forward["five_prime_extension"] = "A" * 200

    with pytest.raises(TriJunctionConfigError, match="recovery forward primer is .*max_oligo_length"):
        parse_request(raw)


def test_load_request_wraps_malformed_yaml(tmp_path: Path) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("targets: [", encoding="utf-8")

    with pytest.raises(TriJunctionConfigError, match="Invalid YAML"):
        load_request(request_path)


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".json", '{"schema":"first","schema":"second"}'),
        (".yaml", "schema: first\nschema: second\n"),
        (".json", '{"planning":{"seed":1,"seed":2}}'),
        (".yaml", "planning:\n  seed: 1\n  seed: 2\n"),
    ],
)
def test_load_request_rejects_duplicate_mapping_keys(tmp_path: Path, suffix: str, content: str) -> None:
    request_path = tmp_path / f"request{suffix}"
    request_path.write_text(content, encoding="utf-8")

    with pytest.raises(TriJunctionConfigError, match="[Dd]uplicate"):
        load_request(request_path)
