"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/ligandmpnn/test_score_results.py

Executed LigandMPNN probability-result boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dnadesign.thread.adapters.ligandmpnn import (
    EXPECTED_LIGANDMPNN_SCORE_ALPHABET,
    LigandMpnnCanonical20Policy,
    LigandMpnnContextAtom,
    LigandMpnnContextInventory,
    LigandMpnnContextInventoryReference,
    LigandMpnnContextPolymer,
    LigandMpnnScoreMode,
    LigandMpnnScoreOutputTrust,
    LigandMpnnScoreRequest,
    LigandMpnnUpstreamPin,
    build_ligandmpnn_score_commands,
    parse_ligandmpnn_score_outputs,
    score_request_sha256,
)

_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"  # pragma: allowlist secret
_CHECKPOINT_SHA256 = "a" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prepare_request(root: Path, *, seeds: tuple[int, ...] = (7, 11)) -> LigandMpnnScoreRequest:
    pdb_payload = b"ATOM      1  N   ALA A   1      0.000   0.000   0.000\n"
    pdb_path = root / "inputs/target.pdb"
    pdb_path.parent.mkdir(parents=True)
    pdb_path.write_bytes(pdb_payload)
    context_inventory = _write_context_inventory(root, pdb_sha256=_sha256(pdb_payload))
    return LigandMpnnScoreRequest(
        request_id="generic_context_probe",
        pdb_path=Path("inputs/target.pdb"),
        pdb_sha256=_sha256(pdb_payload),
        output_dir=Path("outputs/scores"),
        upstream=LigandMpnnUpstreamPin(commit=_COMMIT, checkpoint_sha256=_CHECKPOINT_SHA256),
        context_inventory=context_inventory,
        seeds=seeds,
        batch_size=2,
        number_of_batches=10,
        mode=LigandMpnnScoreMode.SINGLE_AA,
        use_sequence=False,
        use_atom_context=True,
        use_side_chain_context=False,
    )


def _write_context_inventory(
    root: Path,
    *,
    pdb_sha256: str,
) -> LigandMpnnContextInventoryReference:
    atoms = (
        LigandMpnnContextAtom(1, "P", "P", 15, "D", "DC", 12, "", LigandMpnnContextPolymer.DNA),
        LigandMpnnContextAtom(2, "P", "P", 15, "E", "G", 66, "", LigandMpnnContextPolymer.RNA),
    )
    inventory = LigandMpnnContextInventory(
        request_id="generic_context_inventory",
        request_sha256="b" * 64,
        input_path=Path("inputs/target.pdb"),
        input_sha256=pdb_sha256,
        upstream_commit=_COMMIT,
        parser_path=Path("data_utils.py"),
        parser_sha256="c" * 64,
        parser_callable="parse_PDB",
        chains=(),
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=False,
        minimum_nucleotide_atoms=1,
        required_polymer_types=(LigandMpnnContextPolymer.DNA, LigandMpnnContextPolymer.RNA),
        atoms=atoms,
    )
    payload = (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = root / "evidence/context-inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return LigandMpnnContextInventoryReference(
        path=Path("evidence/context-inventory.json"),
        sha256=_sha256(payload),
    )


def _score_payload(
    seed: int,
    *,
    draws: int = 20,
    mode: LigandMpnnScoreMode = LigandMpnnScoreMode.SINGLE_AA,
) -> dict[str, object]:
    residue_names = {0: "A1", 1: "A2", 2: "A3"}
    raw_probabilities = np.full((draws, 3, 21), 0.95 / 20.0, dtype=np.float32)
    raw_probabilities[..., -1] = 0.05
    log_probabilities = np.log(raw_probabilities).astype(np.float32)
    means = np.mean(raw_probabilities, axis=0)
    standard_deviations = np.std(raw_probabilities, axis=0)
    if mode is LigandMpnnScoreMode.SINGLE_AA:
        decoding_order = np.tile(np.arange(3, dtype=np.float32), (draws, 3, 1))
    else:
        decoding_order = np.tile(np.arange(3, dtype=np.int64), (draws, 1))
    return {
        "logits": log_probabilities.copy(),
        "probs": raw_probabilities,
        "log_probs": log_probabilities,
        "decoding_order": decoding_order,
        "native_sequence": np.asarray([0, 1, 20], dtype=np.int64),
        "mask": np.ones(3, dtype=np.float32),
        "chain_mask": np.asarray([1, 0, 1], dtype=np.int64),
        "seed": seed,
        "alphabet": list(EXPECTED_LIGANDMPNN_SCORE_ALPHABET),
        "residue_names": residue_names,
        "sequence": ["A", "C", "X"],
        "mean_of_probs": {
            residue_names[index]: dict(zip(EXPECTED_LIGANDMPNN_SCORE_ALPHABET, means[index], strict=True))
            for index in range(3)
        },
        "std_of_probs": {
            residue_names[index]: dict(zip(EXPECTED_LIGANDMPNN_SCORE_ALPHABET, standard_deviations[index], strict=True))
            for index in range(3)
        },
    }


def _write_output(
    root: Path,
    request: LigandMpnnScoreRequest,
    expected_seed: int,
    **payload_overrides: object,
) -> Path:
    payload = _score_payload(expected_seed, mode=request.mode)
    payload.update(payload_overrides)
    path = root / request.output_dir / f"seed_{expected_seed}" / f"{request.pdb_path.stem}.pt"
    path.parent.mkdir(parents=True)
    torch.save(payload, path)
    return path


def _parse(root: Path, request: LigandMpnnScoreRequest):
    commands = build_ligandmpnn_score_commands(request, checkout_root=Path("/opt/LigandMPNN"))
    return parse_ligandmpnn_score_outputs(
        request,
        commands,
        execution_root=root,
        trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
    )


def test_parser_binds_exact_request_commands_inputs_and_raw_probabilities(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path)
    for seed in request.seeds:
        _write_output(tmp_path, request, seed)

    result = _parse(tmp_path, request)

    assert result.request_sha256 == score_request_sha256(request)
    assert result.input_sha256 == f"sha256:{request.pdb_sha256}"
    assert result.provenance.upstream_commit == _COMMIT
    assert result.provenance.checkpoint_sha256 == f"sha256:{_CHECKPOINT_SHA256}"
    assert result.atom_context_requested is True
    assert result.atom_context_status == "enabled_with_observed_nucleotide_context"
    assert result.context_inventory.effective_nucleotide_atom_count == 2
    assert result.expected_draws_per_seed == 20
    assert [output.seed for output in result.outputs] == [7, 11]
    assert all(output.raw_probabilities.shape == (20, 3, 21) for output in result.outputs)
    assert np.allclose(result.outputs[0].raw_x_probabilities, 0.05)
    assert not result.outputs[0].raw_probabilities.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        result.outputs[0].raw_probabilities.setflags(write=True)
    assert result.outputs[0].artifact_path == Path("outputs/scores/seed_7/target.pt")

    policy = LigandMpnnCanonical20Policy(minimum_canonical_mass=0.90)
    canonical = result.outputs[0].canonical20_probabilities(policy)
    assert canonical.shape == (20, 3, 20)
    assert np.allclose(canonical.sum(axis=-1), 1.0)
    assert not canonical.flags.writeable
    with pytest.raises(ValueError, match="minimum canonical mass"):
        result.outputs[0].canonical20_probabilities(LigandMpnnCanonical20Policy(minimum_canonical_mass=0.96))

    receipt = result.to_dict()
    assert receipt["schema_id"] == "thread.ligandmpnn.score_result"
    assert receipt["schema_version"] == 2
    assert receipt["status"] == "completed_validated"
    assert receipt["input"] == {
        "path": "inputs/target.pdb",
        "sha256": f"sha256:{request.pdb_sha256}",
    }
    assert receipt["outputs"][0]["command_sha256"].startswith("sha256:")
    assert receipt["outputs"][0]["output_sha256"].startswith("sha256:")
    assert "raw_x_probability" in receipt["outputs"][0]
    assert receipt["context"]["atom_context_status"] == "enabled_with_observed_nucleotide_context"
    assert receipt["context"]["inventory_reference"] == request.context_inventory.to_dict()
    assert receipt["context"]["observed_inventory"]["observed"]["effective_nucleotide_atom_count"] == 2


def test_request_digest_is_path_portable_and_context_off_is_explicit(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    context_off = replace(request, use_atom_context=False)
    _write_output(tmp_path, context_off, 7)

    result = _parse(tmp_path, context_off)

    assert result.atom_context_requested is False
    assert result.atom_context_status == "disabled_control_with_observed_nucleotide_context"
    relocated = replace(
        context_off,
        pdb_path=Path("different/host/input.pdb"),
        output_dir=Path("different/host/output"),
    )
    assert score_request_sha256(relocated) == score_request_sha256(context_off)


def test_parser_accepts_the_distinct_autoregressive_decoding_order_shape(tmp_path: Path) -> None:
    request = replace(
        _prepare_request(tmp_path, seeds=(7,)),
        mode=LigandMpnnScoreMode.AUTOREGRESSIVE,
    )
    _write_output(tmp_path, request, 7)

    result = _parse(tmp_path, request)

    assert result.mode == "autoregressive"


def test_parser_fails_closed_on_missing_and_extra_output_files(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path)
    _write_output(tmp_path, request, 7)

    with pytest.raises(ValueError, match="missing expected LigandMPNN score outputs"):
        _parse(tmp_path, request)

    _write_output(tmp_path, request, 11)
    extra = tmp_path / request.output_dir / "seed_99/target.pt"
    extra.parent.mkdir(parents=True)
    torch.save(_score_payload(99), extra)
    with pytest.raises(ValueError, match="unexpected LigandMPNN score outputs"):
        _parse(tmp_path, request)


def test_parser_rejects_symlinked_output_artifacts(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    source = _write_output(tmp_path, request, 7)
    linked = source.with_name("linked.pt")
    linked.symlink_to(source)

    with pytest.raises(ValueError, match="must not be symlinks"):
        _parse(tmp_path, request)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"alphabet": list("XCDEFGHIKLMNPQRSTVWYA")}, "raw alphabet"),
        ({"seed": 999}, "seed"),
        ({"probs": np.ones((19, 3, 21), dtype=np.float32) / 21.0}, "expected 20 draws"),
        ({"extra": "schema drift"}, "unexpected keys"),
    ],
)
def test_parser_rejects_mismatched_payloads(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7, **overrides)

    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, request)


def test_parser_rejects_input_or_context_command_drift(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    (tmp_path / request.pdb_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="input SHA256"):
        _parse(tmp_path, request)

    corrected = replace(request, pdb_sha256=_sha256(b"tampered"))
    commands = build_ligandmpnn_score_commands(corrected, checkout_root=Path("/opt/LigandMPNN"))
    argv = list(commands[0].argv)
    context_index = argv.index("--ligand_mpnn_use_atom_context") + 1
    argv[context_index] = "0"
    drifted = (replace(commands[0], argv=tuple(argv)),)
    with pytest.raises(ValueError, match="commands do not exactly match"):
        parse_ligandmpnn_score_outputs(
            corrected,
            drifted,
            execution_root=tmp_path,
            trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
        )


def test_parser_rejects_missing_or_non_nucleotide_observed_context(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    _write_output(tmp_path, request, 7)
    (tmp_path / request.context_inventory.path).unlink()
    with pytest.raises(ValueError, match="context inventory does not exist"):
        _parse(tmp_path, request)

    inventory_path = tmp_path / request.context_inventory.path
    payload = {
        "schema_id": "thread.ligandmpnn.context_inventory",
        "schema_version": 1,
        "status": "completed_validated",
        "request_id": "generic_context_inventory",
        "request_sha256": f"sha256:{'b' * 64}",
        "input": {"path": "inputs/target.pdb", "sha256": f"sha256:{request.pdb_sha256}"},
        "upstream": {"repository": "https://github.com/dauparas/LigandMPNN", "commit": _COMMIT},
        "parser": {
            "path": "data_utils.py",
            "sha256": f"sha256:{'c' * 64}",
            "callable": "parse_PDB",
            "chains": [],
            "parse_all_atoms": False,
            "parse_atoms_with_zero_occupancy": False,
        },
        "requirements": {"minimum_nucleotide_atoms": 1, "required_polymer_types": []},
        "observed": {
            "effective_nonprotein_atom_count": 1,
            "effective_nucleotide_atom_count": 0,
            "polymer_atom_counts": {"dna": 0, "rna": 0, "other": 1},
            "element_counts": {"ZN": 1},
            "chain_ids": ["Z"],
            "residues": [
                {
                    "chain_id": "Z",
                    "residue_name": "ZN",
                    "residue_number": 1,
                    "insertion_code": "",
                    "polymer_type": "other",
                    "effective_atom_count": 1,
                    "elements": {"ZN": 1},
                }
            ],
            "atoms": [
                {
                    "serial": 1,
                    "atom_name": "ZN",
                    "element": "ZN",
                    "upstream_element_type": 30,
                    "chain_id": "Z",
                    "residue_name": "ZN",
                    "residue_number": 1,
                    "insertion_code": "",
                    "polymer_type": "other",
                }
            ],
        },
    }
    inventory_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    inventory_path.write_bytes(inventory_bytes)
    zero_context = replace(
        request,
        context_inventory=LigandMpnnContextInventoryReference(
            path=request.context_inventory.path,
            sha256=_sha256(inventory_bytes),
        ),
    )
    with pytest.raises(ValueError, match="expected at least 1 effective DNA/RNA context atoms"):
        _parse(tmp_path, zero_context)


def test_parser_requires_explicit_trust_and_still_uses_weights_only_loading(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    commands = build_ligandmpnn_score_commands(request, checkout_root=Path("/opt/LigandMPNN"))
    _write_output(tmp_path, request, 7)

    with pytest.raises(ValueError, match="explicit pinned-local-execution trust"):
        parse_ligandmpnn_score_outputs(request, commands, execution_root=tmp_path, trust="trusted")  # type: ignore[arg-type]

    payload = _score_payload(7)
    payload["untrusted_global"] = Path("not-allowlisted")
    output_path = tmp_path / request.output_dir / "seed_7/target.pt"
    torch.save(payload, output_path)
    with pytest.raises(ValueError, match="weights-only loader rejected"):
        parse_ligandmpnn_score_outputs(
            request,
            commands,
            execution_root=tmp_path,
            trust=LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION,
        )
