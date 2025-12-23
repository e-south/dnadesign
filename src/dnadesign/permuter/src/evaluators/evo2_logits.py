"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/evo2_logits.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List

from dnadesign.permuter.src.evaluators.base import Evaluator


class Evo2LogitsMeanEvaluator(Evaluator):
    def __init__(
        self,
        *,
        model_id: str = "evo2_7b",
        device: str = "cuda:0",
        precision: str = "bf16",
        alphabet: str = "dna",
        batch_size: int | None = 64,
        vocab_reduction: str | None = None,  # optional: "mean"|"sum"|"max"|"min"
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            batch_size=batch_size,
        )
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.vocab_reduction = (vocab_reduction or "").strip().lower() or None
        self._ready = False
        self._rex = None
        self._log = logging.getLogger("permuter.evaluator.evo2_logits")

    def _ensure_ready(self):
        if self._ready:
            return
        if self._rex is None:
            try:
                from dnadesign.infer import run_extract as _rex
            except Exception as e:
                raise RuntimeError(
                    "Evo2 backend unavailable: dnadesign.infer.run_extract is not importable. "
                    "Install evo2 and ensure dnadesign.infer is available."
                ) from e
            self._rex = _rex
        probe_seq = ["ACGTAC"] if self.alphabet.lower().startswith("dna") else ["ACDE"]
        outputs = [
            {
                "id": "logits_mean",
                "fn": "evo2.logits",
                "params": {"pool": {"method": "mean", "dim": 1}},
                "format": "list",
            }
        ]
        try:
            res = self._rex(
                probe_seq,
                model_id=self.model_id,
                outputs=outputs,
                device=self.device,
                precision=self.precision,
                alphabet=self.alphabet,
                batch_size=1,
            )
            probe = res["logits_mean"][0]
            # Accept either a vector [V] (expected) or a scalar, and sanity-check numeric content
            if isinstance(probe, (list, tuple)):
                if not probe:
                    raise RuntimeError("empty logits vector from Evo2")
                _ = float(probe[0])
            else:
                _ = float(probe)
        except Exception as e:
            raise RuntimeError(
                f"Evo2 logits probe failed. Check model_id/device/precision/alphabet and environment. Details: {e}"
            ) from e
        self._ready = True

    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence=None,
        ref_embedding=None,
    ) -> List[float]:
        if str(metric).strip().lower() != "logits_mean":
            raise ValueError("evo2_logits only supports metric='logits_mean'")
        self._ensure_ready()
        seqs = [str(s).upper() for s in sequences]
        outputs = [
            {
                "id": "logits_mean",
                "fn": "evo2.logits",
                "params": {"pool": {"method": "mean", "dim": 1}},
                "format": "list",
            }
        ]
        res = self._rex(
            seqs,
            model_id=self.model_id,
            outputs=outputs,
            device=self.device,
            precision=self.precision,
            alphabet=self.alphabet,
            batch_size=self.batch_size,
        )
        rows = res["logits_mean"]
        # Optional scalar reduction over vocab dimension; otherwise return the [V] vector per sequence
        if self.vocab_reduction is None:
            out = []
            for row in rows:
                if isinstance(row, (list, tuple)):
                    out.append([float(v) for v in row])
                else:
                    out.append(float(row))
            return out
        # Supported reductions: mean, sum, max, min (no SciPy dependency)
        red = self.vocab_reduction
        out = []
        for row in rows:
            vec = row if isinstance(row, (list, tuple)) else [float(row)]
            if red == "mean":
                out.append(float(sum(vec) / max(1, len(vec))))
            elif red == "sum":
                out.append(float(sum(vec)))
            elif red in ("max", "amax"):
                out.append(float(max(vec)))
            elif red in ("min", "amin"):
                out.append(float(min(vec)))
            else:
                raise ValueError(
                    f"Unsupported vocab_reduction={self.vocab_reduction!r}. "
                    "Use one of: mean,sum,max,min; or omit to keep the [V] vector."
                )
        return out
