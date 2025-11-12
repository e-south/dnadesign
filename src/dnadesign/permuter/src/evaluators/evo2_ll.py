"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/evo2_ll.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List

from dnadesign.permuter.src.evaluators.base import Evaluator


class Evo2LogLikelihoodEvaluator(Evaluator):
    def __init__(
        self,
        *,
        model_id: str = "evo2_7b",
        device: str = "cuda:0",
        precision: str = "bf16",
        alphabet: str = "dna",
        method: str = "native",
        reduction: str = "mean",
        batch_size: int | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=device,
            precision=precision,
            alphabet=alphabet,
            method=method,
            reduction=reduction,
            batch_size=batch_size,
        )
        (
            self.model_id,
            self.device,
            self.precision,
            self.alphabet,
            self.method,
            self.reduction,
            self.batch_size,
        ) = (model_id, device, precision, alphabet, method, reduction, batch_size)
        self._ready = False
        self._rex = None 
        self._log = logging.getLogger("permuter.evaluator.evo2_ll")

    def _lazy_rex(self):
        if self._rex is None:
            try:
                from dnadesign.infer import run_extract as _rex
            except Exception as e:
                raise RuntimeError(
                    "Evo2 backend unavailable: dnadesign.infer.run_extract is not importable. "
                    "Ensure the 'evo2' package is installed and compatible with your environment."
                ) from e
            self._rex = _rex
        return self._rex

    def _ensure_ready(self):
        if self._ready:
            return
        rex = self._lazy_rex()

        # Probe with a tiny sequence
        probe_seq = ["ACGTAC"] if self.alphabet.lower().startswith("dna") else ["ACDE"]
        outputs = [
            {
                "id": "ll",
                "fn": "evo2.log_likelihood",
                "params": {"method": self.method, "reduction": self.reduction},
                "format": "float",
            }
        ]
        try:
            res = rex(
                probe_seq,
                model_id=self.model_id,
                outputs=outputs,
                device=self.device,
                precision=self.precision,
                alphabet=self.alphabet,
                batch_size=1,
            )
            _ = float(res["ll"][0])
        except Exception as e:
            raise RuntimeError(
                "Evo2 probe failed. Check that your model_id/device/precision/alphabet are valid "
                f"and your environment can run inference (GPU/CPU availability). Details: {e}"
            ) from e
        self._ready = True

    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: str | None = None,
        ref_embedding=None,
    ) -> List[float]:
        if metric not in ("log_likelihood", "ll"):
            raise ValueError(
                f"evo2_ll only supports metric='log_likelihood' (alias 'll'), got {metric!r}"
            )
        self._ensure_ready()
        seqs = [str(s).upper() for s in sequences]
        outputs = [
            {
                "id": "ll",
                "fn": "evo2.log_likelihood",
                "params": {"method": self.method, "reduction": self.reduction},
                "format": "float",
            }
        ]
        rex = self._lazy_rex()
        res = rex(
            seqs,
            model_id=self.model_id,
            outputs=outputs,
            device=self.device,
            precision=self.precision,
            alphabet=self.alphabet,
            batch_size=self.batch_size,
        )
        return [float(x) for x in res["ll"]]
