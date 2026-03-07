"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/adapters/evo2.py

This module adapts the Arc Institute "Evo 2" DNA language model to the
dnadesign.infer engine.

Primary source (API reference snippets used here):
  - https://github.com/ArcInstitute/evo2
  - README examples (as of Feb 2025):
      from evo2 import Evo2
      model = Evo2("evo2_7b")

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .._logging import get_logger
from ..errors import CapabilityError, ModelLoadError
from ..utils import pool_tensor, to_format

_LOG = get_logger(__name__)

try:
    # Per upstream README, the public class is exposed as `from evo2 import Evo2`.
    from evo2 import Evo2 as _Evo2  # type: ignore
except Exception:  # pragma: no cover
    _Evo2 = None


def _all_equal_lengths(items: Sequence[Sequence[int]]) -> bool:
    """True if all nested sequences have the same length (including empty)."""
    it = iter(items)
    try:
        first_len = len(next(it))
    except StopIteration:
        return True
    return all(len(s) == first_len for s in it)


def _bucket_indices_by_token_length(tokens: Sequence[Sequence[int]]) -> List[List[int]]:
    """Group sequence indices by token length for padding-free batched forwards."""
    buckets: Dict[int, List[int]] = {}
    for index, row in enumerate(tokens):
        buckets.setdefault(len(row), []).append(index)
    return [buckets[length] for length in sorted(buckets)]


def _normalize_pool_config(pool: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if pool is None:
        return None

    method = pool.get("method", "mean")
    try:
        dim = int(pool.get("dim", 1))
    except (TypeError, ValueError) as e:
        raise CapabilityError(f"pool.dim must be an integer for Evo2 extract outputs: {e}")

    if dim < 1:
        raise CapabilityError("pool.dim must be >= 1 for Evo2 extract outputs; dim=1 pools sequence positions.")

    return {"method": method, "dim": dim}


def _pool_batched_tensor(
    t: torch.Tensor,
    *,
    pool: Optional[Dict[str, Any]],
    expected_batch_size: int,
) -> torch.Tensor:
    spec = _normalize_pool_config(pool)
    if spec is None:
        return t

    try:
        pooled = pool_tensor(t, method=spec["method"], dim=spec["dim"])
    except Exception as e:
        raise CapabilityError(
            f"Invalid pooling config for Evo2 outputs: method='{spec['method']}', dim={spec['dim']}: {e}"
        )

    if pooled.size(0) != expected_batch_size:
        raise CapabilityError(
            "Pooling must preserve batch axis for Evo2 extract outputs. "
            "Use pool.dim >= 1 and avoid pooling over batch dimension."
        )
    return pooled


class Evo2Adapter:
    """
    Thin adapter around Evo 2 models.

    Capabilities exposed to the engine:
      - logits        : returns per-token logits or pooled variant.
      - embedding     : intermediate-layer representations (per-token or pooled).
      - log_likelihood: native batched sequence scoring via model.score_sequences().
      - generate      : prompt-based sequence generation.

    The engine constructs this adapter with (model_id, device, precision). We
    instantiate the Evo 2 model by id and (when possible) set the underlying
    torch module to eval mode. All ops also run under torch.inference_mode().
    """

    model_id: str
    device: str
    precision: str
    alphabet_default: str = "dna"

    supports = {
        "logits": True,
        "embedding": True,
        "log_likelihood": True,
        "generate": True,
    }

    def __init__(self, model_id: str, device: str, precision: str) -> None:
        if _Evo2 is None:
            raise ModelLoadError(
                "The 'evo2' package is not installed or failed to import. "
                "Install prerequisites per the Evo 2 README and then `pip install evo2`."
            )
        self.model_id = model_id
        self.device = device
        self.precision = precision

        # Instantiate upstream wrapper
        try:
            model = _Evo2(model_id)
        except Exception as e:  # pragma: no cover
            raise ModelLoadError(f"Failed to instantiate Evo 2('{model_id}'): {e}")

        # ──────────────────────────────────────────────────────────────────────
        # Safe eval-mode handling
        # The Evo2 wrapper itself isn't a torch.nn.Module, so calling .eval() on
        # it raises an AttributeError. Look for a real module inside the wrapper
        # and call eval() there; otherwise proceed silently (we still use
        # torch.inference_mode() for all ops).
        # ──────────────────────────────────────────────────────────────────────
        torch_mod = None
        for attr in ("model", "net", "module", "_model"):
            obj = getattr(model, attr, None)
            if obj is not None and hasattr(obj, "eval") and callable(getattr(obj, "eval")):
                torch_mod = obj
                break

        if torch_mod is not None:
            try:
                torch_mod.eval()
            except Exception as e:  # extremely defensive; shouldn't happen
                _LOG.debug("Evo2 underlying module eval() failed: %s", e)
        else:
            _LOG.debug("Evo2 wrapper exposes no torch module with eval(); proceeding under inference_mode().")

        # Basic tokenizer sanity check (README guarantees presence)
        if not hasattr(model, "tokenizer"):
            raise ModelLoadError("Evo2 model missing tokenizer; unexpected install/runtime state.")

        self.model = model  # keep as-is; upstream handles device placement
        self._torch_module = torch_mod  # optional: might be useful for future hooks

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _autocast_ctx(self):
        """
        Precision context used for CUDA inference. We avoid parameter-dtype
        mutation and rely on autocast for fp16/bf16 compute.
        """
        if self.device.startswith("cuda") and self.precision in {"fp16", "bf16"}:
            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            return torch.autocast("cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _tokenize(self, s: str) -> List[int]:
        """
        Tokenize a single DNA string to Evo 2 token IDs.
        Evo 2 README shows: model.tokenizer.tokenize(sequence) -> list[int]
        """
        return list(self.model.tokenizer.tokenize(s))

    def _tokenize_many(self, seqs: List[str]) -> List[List[int]]:
        return [self._tokenize(s) for s in seqs]

    def _stack_equal_len(self, toks: List[List[int]]) -> torch.Tensor:
        """
        Stack equal-length token lists into a [B, L] int64 tensor on self.device.

        Precondition: all row lengths are equal (checked by caller).
        """
        assert toks, "empty token batch"
        assert _all_equal_lengths(toks), "token lists must have equal length to stack"
        x = torch.tensor(toks, dtype=torch.int64, device=self.device)
        return x

    def _run_extract_batches_by_length(
        self,
        *,
        tokens: List[List[int]],
        pool: Optional[Dict[str, Any]],
        batch_forward: Callable[[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        if not tokens:
            return []

        per_index: List[Optional[torch.Tensor]] = [None] * len(tokens)
        with torch.inference_mode(), self._autocast_ctx():
            for group in _bucket_indices_by_token_length(tokens):
                batch_tokens = [tokens[idx] for idx in group]
                x = self._stack_equal_len(batch_tokens)
                batch_tensor = batch_forward(x)
                batch_tensor = _pool_batched_tensor(
                    batch_tensor,
                    pool=pool,
                    expected_batch_size=len(group),
                )
                for row_index, sample_index in enumerate(group):
                    per_index[sample_index] = batch_tensor[row_index]

        if any(value is None for value in per_index):
            raise CapabilityError("Evo2 extract output assembly failed: missing batch outputs.")
        return [value for value in per_index if value is not None]

    # -------------------------------------------------------------------------
    # Ops
    # -------------------------------------------------------------------------

    def logits(
        self,
        seqs: List[str],
        *,
        pool: Optional[Dict[str, Any]] = None,
        fmt: str,
    ) -> List[Any]:
        """
        Forward pass returning per-token logits (or pooled variant).

        Contract:
          - len(return) == len(seqs).
          - Each element corresponds to one input and is cast to `fmt` via utils.to_format.
          - Without pooling: shape per element is [L, V].
          - With pooling over dim=1 (sequence dim): shape per element is [V] (or as resulting tensor).

        Implementation details:
          - Inputs are grouped by token length and executed as padding-free batches.
        """
        tokens = self._tokenize_many(seqs)

        def _forward_logits(x: torch.Tensor) -> torch.Tensor:
            outputs, _ = self.model(x)
            try:
                return outputs[0]  # [B, L, V]
            except Exception as e:
                raise CapabilityError(f"Evo2 forward returned unexpected structure: {e}")

        logits_by_input = self._run_extract_batches_by_length(
            tokens=tokens,
            pool=pool,
            batch_forward=_forward_logits,
        )
        return [to_format(item, fmt) for item in logits_by_input]

    def embedding(
        self,
        seqs: List[str],
        *,
        layer: Any,
        pool: Optional[Dict[str, Any]] = None,
        fmt: str,
    ) -> List[Any]:
        """
        Extract intermediate-layer embeddings.

        Parameters
        ----------
        layer : str | int
            Evo 2 expects a string layer name using dot-notation, e.g. "blocks.28.mlp.l3"
            (per primary source README).

        pool : dict | None
            Optional pooling over a dimension (commonly dim=1 to average over sequence).

        fmt : {"tensor","numpy","list","float"}
            Output casting choice (delegated to utils.to_format).

        Returns
        -------
        list[Any]
            One embedding per input sequence, each cast to `fmt`. Without pooling the
            per-item shape is [L, D]; with pooling over dim=1 it's [D].
        """
        if not isinstance(layer, str):
            raise CapabilityError(
                "Evo2 embedding expects a string layer name like 'blocks.28.mlp.l3'. "
                "Pass adapter-specific names explicitly to avoid ambiguity."
            )

        tokens = self._tokenize_many(seqs)

        def _forward_embedding(x: torch.Tensor) -> torch.Tensor:
            _, embeddings = self.model(x, return_embeddings=True, layer_names=[layer])
            if layer not in embeddings:
                raise CapabilityError(f"Embedding layer '{layer}' not found in Evo2 response.")
            return embeddings[layer]  # [B, L, D]

        embeddings_by_input = self._run_extract_batches_by_length(
            tokens=tokens,
            pool=pool,
            batch_forward=_forward_embedding,
        )
        return [to_format(item, fmt) for item in embeddings_by_input]

    def log_likelihood(self, seqs: List[str], *, method: str = "native", reduction: str = "sum") -> List[float]:
        """
        Compute (log-)likelihoods using Evo 2's native batch scorer.
        """
        if method != "native":
            raise CapabilityError("Evo2 supports only method='native' in v1.")
        if reduction not in {"sum", "mean"}:
            raise CapabilityError("Evo2 log_likelihood supports reduction='sum' or 'mean' only.")
        red = reduction
        with torch.inference_mode():
            values = self.model.score_sequences(seqs, reduce_method=red)
        return [float(v) for v in values]

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
    ) -> Dict[str, List[Any]]:
        """
        Prompt-based sequence generation via Evo2.

        Parameters follow the dnadesign.infer schema and are translated to Evo2:
          - prompt_seqs <- prompts
          - n_tokens    <- max_new_tokens
          - temperature <- temperature
          - top_k/top_p as provided (optional)
          - seed        : sets torch manual seed (simple, sufficient for repeatability)

        Returns a dict:
          { "gen_seqs": list[str] }

        Note: Evo2 may support additional fields (e.g., per-step scores).
        """
        # Map our schema to Evo2.generate kwargs
        kwargs = {
            "prompt_seqs": prompts,
            "n_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if top_k is not None:
            kwargs["top_k"] = int(top_k)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        # Evo2 README does not show num_return_sequences; if/when supported upstream,
        # this adapter can forward it. For now we enforce 1 to avoid implying support.
        if num_return_sequences not in (None, 1):
            _LOG.warning(
                "Evo2.generate currently documented without 'num_return_sequences'; "
                "ignoring value=%s and returning one sequence per prompt.",
                num_return_sequences,
            )

        if seed is not None:
            # Simple global seed. If upstream exposes a generator argument later,
            # prefer that for isolation.
            torch.manual_seed(int(seed))

        with torch.inference_mode():
            try:
                out = self.model.generate(**kwargs)
            except Exception as e:
                raise CapabilityError(f"Evo2.generate failed: {e}")

        # Upstream returns an object with attribute `.sequences` (README).
        seqs = getattr(out, "sequences", None)
        if seqs is None:
            # Fallback: accept list[str] return if implementation changes.
            if isinstance(out, list) and all(isinstance(x, str) for x in out):
                seqs = out
            else:
                raise CapabilityError("Unexpected return type from Evo2.generate: missing '.sequences' attribute.")

        # Defensive: normalize to plain Python list[str].
        return {"gen_seqs": [str(s) for s in list(seqs)]}
