"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/optimizer/dense_arrays.py

Dense-arrays optimizer adapter and helpers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Protocol

import dense_arrays as da

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerRun:
    optimizer: da.Optimizer
    generator: Iterable
    forbid_each: bool = False


class OptimizerAdapter(Protocol):
    def probe_solver(self, backend: str, *, test_length: int = 10) -> None: ...

    def build(
        self,
        *,
        library: list[str],
        sequence_length: int,
        solver: str | None,
        strategy: str,
        fixed_elements: dict | None,
        strands: str = "double",
        regulator_by_index: list[str] | None = None,
        required_regulators: list[str] | None = None,
        min_count_by_regulator: dict[str, int] | None = None,
        min_required_regulators: int | None = None,
        solver_attempt_timeout_seconds: float | None = None,
        solver_threads: int | None = None,
        extra_label: str | None = None,
    ) -> OptimizerRun: ...


_EXTRA_REGULATOR_LABEL = "__densegen__extra__"


def _normalize_extra_label(extra_label: str | None) -> str:
    if extra_label is None:
        return _EXTRA_REGULATOR_LABEL
    label = str(extra_label).strip()
    if not label:
        raise ValueError("extra_label must be a non-empty string")
    return label


def _normalize_regulator_labels(labels: list[str]) -> list[str]:
    cleaned = []
    for label in labels:
        s = str(label).strip()
        if not s:
            raise ValueError("regulator_by_index labels must be non-empty strings")
        cleaned.append(s)
    return cleaned


def _normalize_motif(motif: str, *, label: str) -> str:
    s = str(motif).strip().upper()
    if not s:
        raise ValueError(f"{label} motifs must be non-empty strings")
    if not set(s).issubset({"A", "C", "G", "T"}):
        raise ValueError(f"{label} motifs must contain only A/C/G/T characters")
    return s


def _apply_regulator_constraints(
    opt: da.Optimizer,
    *,
    regulator_by_index: list[str] | None,
    required_regulators: list[str] | None,
    min_count_by_regulator: dict[str, int] | None,
    min_required_regulators: int | None,
    extra_label: str | None = None,
) -> None:
    if not regulator_by_index:
        return
    if not (required_regulators or min_count_by_regulator or min_required_regulators):
        return

    labels = list(regulator_by_index)
    if len(labels) > len(opt.library):
        raise ValueError(
            f"regulator_by_index length exceeds optimizer library length ({len(labels)} > {len(opt.library)})"
        )
    if len(labels) < len(opt.library):
        labels.extend([_normalize_extra_label(extra_label)] * (len(opt.library) - len(labels)))

    labels = _normalize_regulator_labels(labels)
    label_set = set(labels)

    required = [str(r).strip() for r in (required_regulators or []) if str(r).strip()]
    missing = sorted(set(required) - label_set)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Required regulators missing from optimizer mapping: {preview}")

    if min_count_by_regulator:
        extras = sorted(set(min_count_by_regulator.keys()) - label_set)
        if extras:
            preview = ", ".join(extras[:10])
            raise ValueError(f"min_count_by_regulator includes unknown regulators: {preview}")

    opt.add_regulator_constraints(
        labels,
        required=set(required) if required else None,
        min_count_by_regulator=min_count_by_regulator,
        min_required_regulators=min_required_regulators,
    )


def _apply_solver_controls(
    opt: da.Optimizer,
    *,
    solver_attempt_timeout_seconds: float | None,
    threads: int | None,
) -> None:
    if solver_attempt_timeout_seconds is None and threads is None:
        return
    if solver_attempt_timeout_seconds is not None:
        try:
            solver_attempt_timeout_seconds = float(solver_attempt_timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("solver.solver_attempt_timeout_seconds must be a number of seconds > 0") from exc
        if solver_attempt_timeout_seconds <= 0:
            raise ValueError("solver.solver_attempt_timeout_seconds must be > 0")
    if threads is not None:
        try:
            threads = int(threads)
        except (TypeError, ValueError) as exc:
            raise ValueError("solver.threads must be an integer > 0") from exc
        if threads <= 0:
            raise ValueError("solver.threads must be > 0")
    if not hasattr(opt, "build_model"):
        raise RuntimeError("Optimizer does not expose build_model; cannot apply solver controls.")
    original_build_model = opt.build_model

    def _build_model_with_controls(*args, **kwargs):
        original_build_model(*args, **kwargs)
        model = getattr(opt, "model", None)
        if model is None:
            raise RuntimeError("Solver model not initialized; cannot apply solver controls.")
        if solver_attempt_timeout_seconds is not None:
            if not hasattr(model, "SetTimeLimit"):
                raise RuntimeError("Solver model does not support SetTimeLimit.")
            model.SetTimeLimit(int(max(1, round(solver_attempt_timeout_seconds * 1000))))
        if threads is not None:
            if not hasattr(model, "SetNumThreads"):
                raise RuntimeError("Solver model does not support SetNumThreads.")
            model.SetNumThreads(int(threads))

    opt.build_model = _build_model_with_controls


class DenseArraysAdapter:
    def probe_solver(self, backend: str, *, test_length: int = 10) -> None:
        try:
            dummy = da.Optimizer(library=["AT"], sequence_length=test_length)
            _ = dummy.optimal(solver=backend)
            log.debug("Solver selected: %s", backend)
        except Exception as e:
            raise RuntimeError(
                f"Requested solver '{backend}' failed during probe: {e}\n"
                "Please install/configure this solver or choose another in solver.backend."
            ) from e

    def build(
        self,
        *,
        library: list[str],
        sequence_length: int,
        solver: str | None,
        strategy: str,
        fixed_elements: dict | None,
        strands: str = "double",
        regulator_by_index: list[str] | None = None,
        required_regulators: list[str] | None = None,
        min_count_by_regulator: dict[str, int] | None = None,
        min_required_regulators: int | None = None,
        solver_attempt_timeout_seconds: float | None = None,
        solver_threads: int | None = None,
        extra_label: str | None = None,
    ) -> OptimizerRun:
        if strategy != "approximate" and not solver:
            raise ValueError("solver.backend is required unless strategy=approximate")
        solver_name = solver or ""
        wrapper = DenseArrayOptimizer(
            library=library,
            sequence_length=sequence_length,
            solver=solver_name,
            fixed_elements=fixed_elements,
            strands=strands,
        )
        opt = wrapper.get_optimizer_instance()
        _apply_regulator_constraints(
            opt,
            regulator_by_index=regulator_by_index,
            required_regulators=required_regulators,
            min_count_by_regulator=min_count_by_regulator,
            min_required_regulators=min_required_regulators,
            extra_label=extra_label,
        )
        _apply_solver_controls(
            opt,
            solver_attempt_timeout_seconds=solver_attempt_timeout_seconds,
            threads=solver_threads,
        )
        if strategy == "diverse":
            if not hasattr(opt, "solutions_diverse"):
                raise RuntimeError("dense-arrays does not support solutions_diverse on this install.")
            gen = opt.solutions_diverse(solver=solver_name, solver_options=None)
        elif strategy == "iterate":
            gen = opt.solutions(solver=solver_name, solver_options=None)
        elif strategy == "optimal":

            def _gen():
                start = time.monotonic()
                sol = opt.optimal(solver=solver_name, solver_options=None)
                try:
                    setattr(sol, "_densegen_solve_time_s", time.monotonic() - start)
                except Exception:
                    pass
                yield sol

            gen = _gen()
        elif strategy == "approximate":

            def _gen():
                start = time.monotonic()
                sol = opt.approximate()
                try:
                    setattr(sol, "_densegen_solve_time_s", time.monotonic() - start)
                except Exception:
                    pass
                yield sol

            gen = _gen()
        else:
            raise ValueError(f"Unknown solver strategy: {strategy}")
        if strategy in {"diverse", "iterate"}:
            base_gen = iter(gen)

            def _timed_gen():
                while True:
                    start = time.monotonic()
                    try:
                        sol = next(base_gen)
                    except StopIteration:
                        return
                    try:
                        setattr(sol, "_densegen_solve_time_s", time.monotonic() - start)
                    except Exception:
                        pass
                    yield sol

            gen = _timed_gen()
        return OptimizerRun(optimizer=opt, generator=gen, forbid_each=False)


class DenseArrayOptimizer:
    def __init__(
        self,
        library: list[str],
        sequence_length: int,
        solver: str | None = None,
        fixed_elements: dict | None = None,
        strands: str = "double",
    ):
        valid = {"A", "T", "G", "C"}
        filtered = []
        invalid = []
        for motif in library:
            if not isinstance(motif, str):
                invalid.append(motif)
                continue
            s = motif.strip().upper()
            if not s:
                invalid.append(motif)
                continue
            if not set(s).issubset(valid):
                invalid.append(motif)
                continue
            filtered.append(s)
        if invalid:
            preview = ", ".join(repr(x) for x in invalid[:5])
            raise ValueError(f"Invalid motifs in library (showing up to 5): {preview}")
        if not filtered:
            raise ValueError("Motif library is empty after validation.")

        self.library = filtered
        self.sequence_length = sequence_length
        self.solver = solver
        self.fixed_elements = (fixed_elements or {}).copy()
        if strands not in {"single", "double"}:
            raise ValueError("strands must be 'single' or 'double'")
        self.strands = strands

    def get_optimizer_instance(self) -> da.Optimizer:
        lib = self.library.copy()
        converted = []
        cons = (self.fixed_elements or {}).get("promoter_constraints")
        if cons:
            if not isinstance(cons, list):
                raise ValueError("promoter_constraints must be a list of dicts")
            for pc in cons:
                if not isinstance(pc, dict):
                    raise ValueError("Each promoter constraint must be a dict")
                unknown = set(pc.keys()) - {
                    "name",
                    "upstream",
                    "downstream",
                    "upstream_variant_id",
                    "downstream_variant_id",
                    "spacer_length",
                    "upstream_pos",
                    "downstream_pos",
                }
                if unknown:
                    raise ValueError(f"Unknown promoter constraint keys: {sorted(unknown)}")
                for id_key in ("upstream_variant_id", "downstream_variant_id"):
                    id_value = pc.get(id_key)
                    if id_value is None:
                        continue
                    if not isinstance(id_value, str) or not id_value.strip():
                        raise ValueError(f"{id_key} must be a non-empty string when provided")
                # Ensure motifs in library
                clean: dict[str, object] = {}
                for mkey in ("upstream", "downstream"):
                    mv = pc.get(mkey)
                    if mv is None:
                        continue
                    mv_norm = _normalize_motif(mv, label=f"promoter_constraints.{mkey}")
                    if mv_norm not in lib:
                        lib.append(mv_norm)
                    clean[mkey] = mv_norm
                for k in ("spacer_length", "upstream_pos", "downstream_pos"):
                    if k in pc and pc[k] is not None:
                        clean[k] = pc[k]
                if clean:
                    converted.append(clean)

        opt = da.Optimizer(
            library=lib,
            sequence_length=self.sequence_length,
            strands=self.strands,
        )
        for c in converted:
            opt.add_promoter_constraints(**c)

        sb = (self.fixed_elements or {}).get("side_biases") or {}
        left_raw, right_raw = sb.get("left"), sb.get("right")
        left = [_normalize_motif(m, label="side_biases.left") for m in (left_raw or []) if m is not None]
        right = [_normalize_motif(m, label="side_biases.right") for m in (right_raw or []) if m is not None]
        if (left and any(left)) or (right and any(right)):
            missing = [m for m in (left or []) + (right or []) if m not in lib]
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"side_biases motifs must exist in the library: {preview}")
            opt.add_side_biases(left=left, right=right)
        return opt
