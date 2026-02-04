"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/adapters/protocols.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class SupportsExtract(Protocol):
    alphabet_default: str
    supports: Dict[str, bool]

    def logits(self, seqs: List[str], *, pool: Optional[dict], fmt: str) -> List[Any]: ...
    def embedding(self, seqs: List[str], *, layer: str, pool: Optional[dict], fmt: str) -> List[Any]: ...
    def log_likelihood(self, seqs: List[str], *, method: str, reduction: str) -> List[float]: ...


class SupportsGenerate(Protocol):
    def generate(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, List[Any]]: ...
