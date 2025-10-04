"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/evo2_llr.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List

try:
    from dnadesign.infer import run_extract
except Exception:
    run_extract = None
from dnadesign.permuter.src.evaluators.base import Evaluator


class Evo2LogLikelihoodRatioEvaluator(Evaluator):
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
        self._log = logging.getLogger("permuter.evaluator.evo2_llr")

    def _ensure_ready(self):
        if self._ready:
            return
        if run_extract is None:
            raise RuntimeError(
                "Evo2 backend unavailable: dnadesign.infer.run_extract is not importable. "
                "Ensure the 'evo2' package is installed (pip install evo2) and that dnadesign is importable."

            )
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
            res = run_extract(
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
                "Evo2 probe failed. Confirm model_id/device/precision/alphabet and environment. "
                f"Details: {e}"
            ) from e
        self._ready = True

    def _ll(self, seqs: List[str]) -> List[float]:
        if run_extract is None:
            raise RuntimeError(
                "dnadesign.infer.run_extract is unavailable. Install dnadesign with the 'infer-evo2' extra."
            )
        outputs = [
            {
                "id": "ll",
                "fn": "evo2.log_likelihood",
                "params": {"method": self.method, "reduction": self.reduction},
                "format": "float",
            }
        ]
        res = run_extract(
            [str(s).upper() for s in seqs],
            model_id=self.model_id,
            outputs=outputs,
            device=self.device,
            precision=self.precision,
            alphabet=self.alphabet,
            batch_size=self.batch_size,
        )
        return [float(x) for x in res["ll"]]

    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: str | None = None,
        ref_embedding=None,
    ) -> List[float]:
        if metric not in ("log_likelihood_ratio", "llr"):
            raise ValueError(
                f"evo2_llr only supports metric='log_likelihood_ratio' (alias 'llr'), got {metric!r}"
            )
        if not ref_sequence:
            raise ValueError("evo2_llr requires ref_sequence")
        self._ensure_ready()
        ll_variants = self._ll(sequences)
        ll_ref = self._ll([ref_sequence])[0]
        return [v - ll_ref for v in ll_variants]
