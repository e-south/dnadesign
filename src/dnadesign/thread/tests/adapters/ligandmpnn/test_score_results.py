"""Executed LigandMPNN probability-result boundary tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dnadesign.thread.adapters.ligandmpnn import (
    EXPECTED_LIGANDMPNN_SCORE_ALPHABET,
    LigandMpnnCanonical20Policy,
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
    return LigandMpnnScoreRequest(
        request_id="generic_context_probe",
        pdb_path=Path("inputs/target.pdb"),
        pdb_sha256=_sha256(pdb_payload),
        output_dir=Path("outputs/scores"),
        upstream=LigandMpnnUpstreamPin(commit=_COMMIT, checkpoint_sha256=_CHECKPOINT_SHA256),
        seeds=seeds,
        batch_size=2,
        number_of_batches=10,
        mode=LigandMpnnScoreMode.SINGLE_AA,
        use_sequence=False,
        use_atom_context=True,
        use_side_chain_context=False,
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
    assert result.atom_context == "on"
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
    assert receipt["status"] == "completed_validated"
    assert receipt["input"] == {
        "path": "inputs/target.pdb",
        "sha256": f"sha256:{request.pdb_sha256}",
    }
    assert receipt["outputs"][0]["command_sha256"].startswith("sha256:")
    assert receipt["outputs"][0]["output_sha256"].startswith("sha256:")
    assert "raw_x_probability" in receipt["outputs"][0]


def test_request_digest_is_path_portable_and_context_off_is_explicit(tmp_path: Path) -> None:
    request = _prepare_request(tmp_path, seeds=(7,))
    context_off = replace(request, use_atom_context=False)
    _write_output(tmp_path, context_off, 7)

    result = _parse(tmp_path, context_off)

    assert result.atom_context == "off"
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
