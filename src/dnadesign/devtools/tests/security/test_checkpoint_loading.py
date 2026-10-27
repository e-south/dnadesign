"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/security/test_checkpoint_loading.py

Runtime checks for restricted checkpoint deserialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest


class _UnsupportedCheckpointValue:
    pass


def test_restricted_torch_load_rejects_unsupported_globals(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "unsupported.pt"
    torch.save([{"sequence": "ACGT", "value": _UnsupportedCheckpointValue()}], path)

    with pytest.raises((pickle.UnpicklingError, ValueError), match="Weights only load failed"):
        torch.load(path, map_location="cpu", weights_only=True)


def test_restricted_torch_load_preserves_plain_tensor_and_dict_payloads(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "plain.pt"
    payload = [{"sequence": "ACGT", "embedding": torch.tensor([1.0, 2.0])}]
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded[0]["sequence"] == "ACGT"
    assert loaded[0]["embedding"].tolist() == [1.0, 2.0]
