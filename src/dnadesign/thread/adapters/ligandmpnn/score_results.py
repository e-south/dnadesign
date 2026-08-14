"""Strict parsing and lineage for official LigandMPNN ``score.py`` outputs.

The pinned upstream writes NumPy arrays through ``torch.save``. PyTorch files
are pickle-capable containers, so this module accepts them only from an
explicitly attested, pinned local execution root. It still uses
``weights_only=True`` with a narrow NumPy allowlist. That reduces the loading
surface; it does not make arbitrary downloaded ``.pt`` files trustworthy.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dnadesign.thread.adapters.ligandmpnn.models import LigandMpnnCommand
from dnadesign.thread.adapters.ligandmpnn.receipts import (
    LigandMpnnProvenance,
)
from dnadesign.thread.adapters.ligandmpnn.scoring import (
    LigandMpnnScoreMode,
    LigandMpnnScoreRequest,
    build_ligandmpnn_score_commands,
)

EXPECTED_LIGANDMPNN_SCORE_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_EXPECTED_KEYS = frozenset(
    {
        "logits",
        "probs",
        "log_probs",
        "decoding_order",
        "native_sequence",
        "mask",
        "chain_mask",
        "seed",
        "alphabet",
        "residue_names",
        "sequence",
        "mean_of_probs",
        "std_of_probs",
    }
)


class LigandMpnnScoreOutputTrust(str, Enum):
    """Caller attestation accepted by the PyTorch output boundary."""

    PINNED_LOCAL_EXECUTION = "pinned_local_execution"


@dataclass(frozen=True)
class LigandMpnnCanonical20Policy:
    """Explicit caller policy for conditioning probabilities on non-``X``.

    ``minimum_canonical_mass`` is a numerical guard chosen by the caller. The
    adapter does not choose a biological eligibility threshold or discard
    observations based on ``pX``.
    """

    minimum_canonical_mass: float

    def __post_init__(self) -> None:
        value = self.minimum_canonical_mass
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("minimum_canonical_mass must be finite and in (0, 1]")
        if not math.isfinite(value) or value <= 0 or value > 1:
            raise ValueError("minimum_canonical_mass must be finite and in (0, 1]")


@dataclass(frozen=True)
class LigandMpnnScoreOutput:
    """One validated per-seed upstream probability artifact."""

    seed: int
    artifact_path: Path
    output_sha256: str
    command_sha256: str
    residue_names: tuple[str, ...]
    _raw_probabilities: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        probabilities = np.array(self._raw_probabilities, copy=True)
        probabilities.setflags(write=False)
        object.__setattr__(self, "_raw_probabilities", probabilities)

    @property
    def draw_count(self) -> int:
        return int(self._raw_probabilities.shape[0])

    @property
    def residue_count(self) -> int:
        return int(self._raw_probabilities.shape[1])

    @property
    def raw_probabilities(self) -> np.ndarray:
        """Return the immutable raw 21-state probabilities, including ``X``."""

        values = self._raw_probabilities.view()
        values.setflags(write=False)
        return values

    @property
    def raw_x_probabilities(self) -> np.ndarray:
        """Return the immutable raw upstream probability assigned to ``X``."""

        values = self._raw_probabilities[..., -1].view()
        values.setflags(write=False)
        return values

    def canonical20_probabilities(self, policy: LigandMpnnCanonical20Policy) -> np.ndarray:
        """Condition on a canonical residue only under an explicit caller policy."""

        if not isinstance(policy, LigandMpnnCanonical20Policy):
            raise ValueError("policy must be a LigandMpnnCanonical20Policy")
        canonical = self._raw_probabilities[..., :20]
        canonical_mass = canonical.sum(axis=-1)
        observed_minimum = float(canonical_mass.min())
        if observed_minimum < policy.minimum_canonical_mass:
            raise ValueError(
                "minimum canonical mass "
                f"{observed_minimum:.8g} is below caller policy {policy.minimum_canonical_mass:.8g}"
            )
        normalized = np.asarray(canonical / canonical_mass[..., np.newaxis])
        normalized.setflags(write=False)
        values = normalized.view()
        values.setflags(write=False)
        return values

    def to_dict(self) -> dict[str, object]:
        x_probabilities = self.raw_x_probabilities
        return {
            "seed": self.seed,
            "artifact_path": self.artifact_path.as_posix(),
            "output_sha256": self.output_sha256,
            "command_sha256": self.command_sha256,
            "draw_count": self.draw_count,
            "residue_count": self.residue_count,
            "raw_alphabet": EXPECTED_LIGANDMPNN_SCORE_ALPHABET,
            "raw_x_probability": {
                "minimum": float(x_probabilities.min()),
                "maximum": float(x_probabilities.max()),
                "mean": float(x_probabilities.mean()),
            },
        }


@dataclass(frozen=True)
class LigandMpnnScoreResult:
    """Validated request-to-result lineage for one score request."""

    request_id: str
    request_sha256: str
    input_path: Path
    input_sha256: str
    command_set_sha256: str
    provenance: LigandMpnnProvenance
    mode: str
    atom_context: str
    side_chain_context: str
    use_sequence: bool
    batch_size: int
    number_of_batches: int
    outputs: tuple[LigandMpnnScoreOutput, ...]

    @property
    def expected_draws_per_seed(self) -> int:
        return self.batch_size * self.number_of_batches

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": "thread.ligandmpnn.score_result",
            "schema_version": 1,
            "status": "completed_validated",
            "model_type": "ligand_mpnn",
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "input": {"path": self.input_path.as_posix(), "sha256": self.input_sha256},
            "command_set_sha256": self.command_set_sha256,
            "provenance": self.provenance.to_dict(),
            "score_mode": self.mode,
            "context": {
                "atom_context": self.atom_context,
                "side_chain_context": self.side_chain_context,
                "use_sequence": self.use_sequence,
            },
            "batch_size": self.batch_size,
            "number_of_batches": self.number_of_batches,
            "expected_draws_per_seed": self.expected_draws_per_seed,
            "expected_output_count": len(self.outputs),
            "validated_output_count": len(self.outputs),
            "outputs": [output.to_dict() for output in self.outputs],
        }


def score_request_sha256(request: LigandMpnnScoreRequest) -> str:
    """Return a path-portable digest of one semantic score request."""

    payload = {
        "schema_id": "thread.ligandmpnn.score_request",
        "schema_version": 1,
        "request_id": request.request_id,
        "pdb_sha256": f"sha256:{request.pdb_sha256}",
        "upstream": {
            "repository": LigandMpnnProvenance.from_pin(request.upstream).upstream_repository,
            "commit": request.upstream.commit,
            "checkpoint_path": request.upstream.checkpoint_path.as_posix(),
            "checkpoint_sha256": f"sha256:{request.upstream.checkpoint_sha256}",
        },
        "fixed_residues": [residue.upstream_id for residue in request.fixed_residues],
        "redesigned_residues": [residue.upstream_id for residue in request.redesigned_residues],
        "seeds": list(request.seeds),
        "batch_size": request.batch_size,
        "number_of_batches": request.number_of_batches,
        "mode": request.mode.value,
        "use_sequence": request.use_sequence,
        "use_atom_context": request.use_atom_context,
        "use_side_chain_context": request.use_side_chain_context,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def parse_ligandmpnn_score_outputs(
    request: LigandMpnnScoreRequest,
    commands: tuple[LigandMpnnCommand, ...],
    *,
    execution_root: Path,
    trust: LigandMpnnScoreOutputTrust,
) -> LigandMpnnScoreResult:
    """Parse exactly one pinned upstream score artifact per requested seed.

    The caller must attest that ``execution_root`` is the local output root of
    the pinned commands. Files copied from untrusted sources are outside this
    contract even though the loader also uses PyTorch's restricted mode.
    """

    if trust is not LigandMpnnScoreOutputTrust.PINNED_LOCAL_EXECUTION:
        raise ValueError("score parsing requires explicit pinned-local-execution trust")
    root = execution_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("execution_root must be an existing directory")

    input_path = _within_root(root, request.pdb_path, field_name="pdb_path")
    if not input_path.is_file():
        raise ValueError(f"score input does not exist: {request.pdb_path}")
    observed_input_sha256 = _sha256_file(input_path)
    expected_input_sha256 = f"sha256:{request.pdb_sha256}"
    if observed_input_sha256 != expected_input_sha256:
        raise ValueError(f"input SHA256 mismatch: expected {expected_input_sha256}, observed {observed_input_sha256}")

    _validate_commands(request, commands)
    command_digests = tuple(_command_sha256(command) for command in commands)
    command_set_sha256 = _sha256_bytes(json.dumps(command_digests, separators=(",", ":")).encode("utf-8"))

    output_root = _within_root(root, request.output_dir, field_name="output_dir")
    expected_paths = tuple(
        _within_root(root, command.output_dir / f"{request.pdb_path.stem}.pt", field_name="score output")
        for command in commands
    )
    discovered_paths = tuple(output_root.rglob("*.pt")) if output_root.is_dir() else ()
    symlink_outputs = tuple(path for path in discovered_paths if path.is_symlink())
    if symlink_outputs:
        raise ValueError(
            "LigandMPNN score outputs must not be symlinks: "
            + ", ".join(path.relative_to(root).as_posix() for path in symlink_outputs)
        )
    observed_paths = {path.resolve() for path in discovered_paths if path.is_file()}
    expected_set = set(expected_paths)
    missing = sorted(expected_set - observed_paths)
    extra = sorted(observed_paths - expected_set)
    if missing:
        raise ValueError(
            "missing expected LigandMPNN score outputs: "
            + ", ".join(path.relative_to(root).as_posix() for path in missing)
        )
    if extra:
        raise ValueError(
            "unexpected LigandMPNN score outputs: " + ", ".join(_display_path(path, root) for path in extra)
        )

    outputs: list[LigandMpnnScoreOutput] = []
    for command, command_sha256, output_path in zip(commands, command_digests, expected_paths, strict=True):
        payload_bytes = output_path.read_bytes()
        payload = _load_weights_only_payload(payload_bytes, artifact_path=output_path.relative_to(root))
        residue_names, raw_probabilities = _validate_payload(
            payload,
            seed=command.seed,
            expected_draws=request.batch_size * request.number_of_batches,
            mode=request.mode,
        )
        outputs.append(
            LigandMpnnScoreOutput(
                seed=command.seed,
                artifact_path=output_path.relative_to(root),
                output_sha256=_sha256_bytes(payload_bytes),
                command_sha256=command_sha256,
                residue_names=residue_names,
                _raw_probabilities=raw_probabilities,
            )
        )

    return LigandMpnnScoreResult(
        request_id=request.request_id,
        request_sha256=score_request_sha256(request),
        input_path=input_path.relative_to(root),
        input_sha256=observed_input_sha256,
        command_set_sha256=command_set_sha256,
        provenance=LigandMpnnProvenance.from_pin(request.upstream),
        mode=request.mode.value,
        atom_context="on" if request.use_atom_context else "off",
        side_chain_context="on" if request.use_side_chain_context else "off",
        use_sequence=request.use_sequence,
        batch_size=request.batch_size,
        number_of_batches=request.number_of_batches,
        outputs=tuple(outputs),
    )


def _validate_commands(request: LigandMpnnScoreRequest, commands: tuple[LigandMpnnCommand, ...]) -> None:
    if not isinstance(commands, tuple) or len(commands) != len(request.seeds):
        raise ValueError("commands do not exactly match score request seeds")
    if not commands or len(commands[0].argv) < 2:
        raise ValueError("commands do not exactly match score request")
    first_argv = commands[0].argv
    checkout_root = Path(first_argv[1]).parent
    expected = build_ligandmpnn_score_commands(
        request,
        checkout_root=checkout_root,
        python_executable=first_argv[0],
    )
    if commands != expected:
        raise ValueError("commands do not exactly match score request and context mode")


def _load_weights_only_payload(payload: bytes, *, artifact_path: Path) -> dict[str, Any]:
    safe_globals = [
        np._core.multiarray._reconstruct,  # noqa: SLF001
        np._core.multiarray.scalar,  # noqa: SLF001
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.bool_)),
    ]
    try:
        with torch.serialization.safe_globals(safe_globals):
            loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"weights-only loader rejected {artifact_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"score output {artifact_path} must contain a dictionary")
    return loaded


def _validate_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    expected_draws: int,
    mode: LigandMpnnScoreMode,
) -> tuple[tuple[str, ...], np.ndarray]:
    keys = set(payload)
    missing = sorted(_EXPECTED_KEYS - keys)
    extra = sorted(keys - _EXPECTED_KEYS)
    if missing:
        raise ValueError(f"score output is missing keys: {', '.join(missing)}")
    if extra:
        raise ValueError(f"score output has unexpected keys: {', '.join(extra)}")
    if payload["alphabet"] != list(EXPECTED_LIGANDMPNN_SCORE_ALPHABET):
        raise ValueError(f"score output raw alphabet must be exactly {EXPECTED_LIGANDMPNN_SCORE_ALPHABET}")
    observed_seed = payload["seed"]
    if isinstance(observed_seed, bool) or not isinstance(observed_seed, int) or observed_seed != seed:
        raise ValueError(f"score output seed {observed_seed!r} does not match expected seed {seed}")

    probabilities = _numeric_array(payload, "probs", dimensions=3)
    if probabilities.shape[0] != expected_draws:
        raise ValueError(f"score output probs has {probabilities.shape[0]} draws; expected {expected_draws} draws")
    if probabilities.shape[2] != len(EXPECTED_LIGANDMPNN_SCORE_ALPHABET):
        raise ValueError("score output probs final dimension must match the raw 21-state alphabet")
    residue_count = probabilities.shape[1]
    if residue_count <= 0:
        raise ValueError("score output must contain at least one protein residue")
    expected_probability_shape = (expected_draws, residue_count, 21)
    _numeric_array(payload, "logits", shape=expected_probability_shape)
    log_probabilities = _numeric_array(payload, "log_probs", shape=expected_probability_shape)
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("score output probs must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("score output probs must sum to one across the raw alphabet")
    if not np.allclose(np.exp(log_probabilities), probabilities, rtol=1e-5, atol=1e-7):
        raise ValueError("score output probs do not match exp(log_probs)")
    _validate_decoding_order(
        payload,
        mode=mode,
        expected_draws=expected_draws,
        residue_count=residue_count,
    )
    native_sequence = _integer_array(payload, "native_sequence", shape=(residue_count,))
    if np.any(native_sequence < 0) or np.any(native_sequence >= 21):
        raise ValueError("score output native_sequence contains an out-of-range alphabet index")
    _binary_array(payload, "mask", shape=(residue_count,))
    _binary_array(payload, "chain_mask", shape=(residue_count,))

    residue_names = _residue_names(payload["residue_names"], residue_count=residue_count)
    sequence = payload["sequence"]
    expected_sequence = [EXPECTED_LIGANDMPNN_SCORE_ALPHABET[index] for index in native_sequence]
    if sequence != expected_sequence:
        raise ValueError("score output sequence does not match native_sequence and raw alphabet")
    _validate_summary(
        payload["mean_of_probs"],
        expected=probabilities.mean(axis=0),
        residue_names=residue_names,
        field_name="mean_of_probs",
    )
    _validate_summary(
        payload["std_of_probs"],
        expected=probabilities.std(axis=0),
        residue_names=residue_names,
        field_name="std_of_probs",
    )
    return residue_names, np.asarray(probabilities)


def _numeric_array(
    payload: dict[str, Any],
    field_name: str,
    *,
    dimensions: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    value = payload[field_name]
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"score output {field_name} must be a numeric NumPy array")
    if dimensions is not None and value.ndim != dimensions:
        raise ValueError(f"score output {field_name} must have {dimensions} dimensions")
    if shape is not None and value.shape != shape:
        raise ValueError(f"score output {field_name} shape {value.shape} does not match expected {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"score output {field_name} must be finite")
    return value


def _integer_array(payload: dict[str, Any], field_name: str, *, shape: tuple[int, ...]) -> np.ndarray:
    value = _numeric_array(payload, field_name, shape=shape)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"score output {field_name} must use an integer dtype")
    return value


def _validate_decoding_order(
    payload: dict[str, Any],
    *,
    mode: LigandMpnnScoreMode,
    expected_draws: int,
    residue_count: int,
) -> None:
    if mode is LigandMpnnScoreMode.SINGLE_AA:
        value = _numeric_array(
            payload,
            "decoding_order",
            shape=(expected_draws, residue_count, residue_count),
        )
        if not np.equal(value, np.floor(value)).all():
            raise ValueError("score output single-AA decoding_order must contain integer-valued indices")
    else:
        value = _integer_array(payload, "decoding_order", shape=(expected_draws, residue_count))
    if np.any(value < 0) or np.any(value >= residue_count):
        raise ValueError("score output decoding_order contains an out-of-range residue index")
    expected_permutation = np.arange(residue_count)
    if not np.equal(np.sort(value, axis=-1), expected_permutation).all():
        raise ValueError("score output decoding_order rows must each be a complete residue permutation")


def _binary_array(payload: dict[str, Any], field_name: str, *, shape: tuple[int, ...]) -> np.ndarray:
    value = _numeric_array(payload, field_name, shape=shape)
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"score output {field_name} must contain only zero or one")
    return value


def _residue_names(value: object, *, residue_count: int) -> tuple[str, ...]:
    if not isinstance(value, dict) or set(value) != set(range(residue_count)):
        raise ValueError("score output residue_names must map each zero-based residue index exactly once")
    names = tuple(value[index] for index in range(residue_count))
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != residue_count:
        raise ValueError("score output residue_names must contain unique nonempty strings")
    return names


def _validate_summary(
    value: object,
    *,
    expected: np.ndarray,
    residue_names: tuple[str, ...],
    field_name: str,
) -> None:
    if not isinstance(value, dict) or set(value) != set(residue_names):
        raise ValueError(f"score output {field_name} must contain every residue exactly once")
    for index, residue_name in enumerate(residue_names):
        row = value[residue_name]
        if not isinstance(row, dict) or list(row) != list(EXPECTED_LIGANDMPNN_SCORE_ALPHABET):
            raise ValueError(f"score output {field_name} row {residue_name} has alphabet drift")
        observed = np.asarray(list(row.values()), dtype=np.float64)
        if not np.isfinite(observed).all() or not np.allclose(observed, expected[index], rtol=1e-5, atol=1e-7):
            raise ValueError(f"score output {field_name} row {residue_name} does not match raw probabilities")


def _within_root(root: Path, path: Path, *, field_name: str) -> Path:
    if not isinstance(path, Path):
        raise ValueError(f"{field_name} must be a Path")
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} must resolve within execution_root") from error
    return resolved


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _command_sha256(command: LigandMpnnCommand) -> str:
    payload = json.dumps(command.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
