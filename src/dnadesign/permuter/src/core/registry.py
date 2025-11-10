"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/registry.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Dict, Type

# caches
_PROTO: dict[str, Type] = {}
_EVAL: dict[str, Type] = {}
_LOG = logging.getLogger("permuter.registry")


def _load(group: str) -> dict[str, Type]:
    out: dict[str, Type] = {}
    try:
        eps = entry_points().select(group=f"permuter.{group}")
        names = [ep.name for ep in eps]
        _LOG.debug(f"[registry] discovered entry points for '{group}': {names or []}")
        for ep in eps:
            try:
                out[ep.name] = ep.load()
                _LOG.debug(f"[registry] loaded entry point '{group}:{ep.name}'")
            except Exception as e:
                _LOG.debug(f"[registry] failed to load '{group}:{ep.name}': {e}")
    except Exception as e:
        _LOG.debug(f"[registry] entry point scan failed for '{group}': {e}")
    return out


def _builtins(group: str) -> dict[str, Type]:
    """
    Load built-ins. Core items (protocol: scan_dna; evaluator: placeholder)
    are hard requirements; optional ones are best-effort with DEBUG logs.
    """
    out: Dict[str, Type] = {}
    if group == "protocols":
        try:
            from dnadesign.permuter.src.protocols.dms.scan_dna import ScanDNA
        except Exception as e:
            # core protocol must be present
            raise RuntimeError(
                f"[registry] core builtin protocol 'scan_dna' failed to import: {e}"
            ) from e
        out["scan_dna"] = ScanDNA
        try:
            from dnadesign.permuter.src.protocols.dms.scan_codon import ScanCodon

            out["scan_codon"] = ScanCodon
        except Exception as e:
            _LOG.debug(
                f"[registry] optional builtin protocol 'scan_codon' not available: {e}"
            )
        try:
            from dnadesign.permuter.src.protocols.hairpins.scan_stem_loop import (
                ScanStemLoop,
            )

            out["scan_stem_loop"] = ScanStemLoop
        except Exception as e:
            _LOG.debug(
                f"[registry] optional builtin protocol 'scan_stem_loop' not available: {e}"
            )
        try:
            from dnadesign.permuter.src.protocols.combine.combine_aa import CombineAA
            out["combine_aa"] = CombineAA
        except Exception as e:
            _LOG.debug(
                f"[registry] optional builtin protocol 'combine_aa' not available: {e}"
            )
        return out
    else:
        try:
            from dnadesign.permuter.src.evaluators.placeholder import (
                PlaceholderEvaluator,
            )
        except Exception as e:
            # core evaluator must be present
            raise RuntimeError(
                f"[registry] core builtin evaluator 'placeholder' failed to import: {e}"
            ) from e
        out["placeholder"] = PlaceholderEvaluator
        try:
            from dnadesign.permuter.src.evaluators.evo2_ll import (
                Evo2LogLikelihoodEvaluator,
            )

            out["evo2_ll"] = Evo2LogLikelihoodEvaluator
            from dnadesign.permuter.src.evaluators.evo2_llr import (
                Evo2LogLikelihoodRatioEvaluator,
            )

            out["evo2_llr"] = Evo2LogLikelihoodRatioEvaluator
        except Exception as e:
            _LOG.debug(
                f"[registry] optional builtin evaluators 'evo2_ll/llr' not available: {e}"
            )
        return out


def get_protocol(name: str) -> Type:
    if not _PROTO:
        # Load plugins then ensure built-ins (core guaranteed or raise).
        _PROTO.update(_load("protocols"))
        _PROTO.update(_builtins("protocols"))
        _LOG.debug(f"[registry] available protocols: {sorted(_PROTO.keys())}")
    if name in _PROTO:
        return _PROTO[name]
    available = ", ".join(sorted(_PROTO.keys()))
    raise ValueError(f"Unknown protocol: {name!r}. Available: [{available}]")


def get_evaluator(name: str) -> Type:
    if not _EVAL:
        _EVAL.update(_load("evaluators"))
        _EVAL.update(_builtins("evaluators"))
        _LOG.debug(f"[registry] available evaluators: {sorted(_EVAL.keys())}")
    if name in _EVAL:
        return _EVAL[name]
    available = ", ".join(sorted(_EVAL.keys()))
    raise ValueError(f"Unknown evaluator: {name!r}. Available: [{available}]")
