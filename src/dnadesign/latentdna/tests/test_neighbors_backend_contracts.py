"""
Neighbor backend contract tests for latentdna.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from dnadesign.latentdna.src.neighbors.backends.approximate import fit_neighbors_approximate


def test_fit_neighbors_approximate_uses_training_neighbor_graph(monkeypatch) -> None:
    class FakeNNDescent:
        def __init__(self, data, **kwargs) -> None:
            del kwargs
            assert data.flags.c_contiguous
            assert data.flags.writeable
            row_count = int(data.shape[0])
            self._indices = np.asarray(
                [
                    [row_index, (row_index + 1) % row_count, (row_index + 2) % row_count]
                    for row_index in range(row_count)
                ],
                dtype=np.int64,
            )
            self._distances = np.asarray(
                [[0.0, 0.1 + row_index, 0.2 + row_index] for row_index in range(row_count)],
                dtype=np.float32,
            )

        @property
        def neighbor_graph(self) -> tuple[np.ndarray, np.ndarray]:
            return self._indices, self._distances

        def query(self, *args, **kwargs):
            raise AssertionError("fit_neighbors_approximate should not re-query the training matrix")

    monkeypatch.setitem(sys.modules, "pynndescent", SimpleNamespace(NNDescent=FakeNNDescent))

    matrix = np.arange(15, dtype=np.float32).reshape(5, 3)
    matrix.setflags(write=False)
    indices, distances = fit_neighbors_approximate(matrix, k=2, metric="euclidean", seed=17)

    assert indices.shape == (5, 2)
    assert distances.shape == (5, 2)
    assert indices.tolist()[0] == [1, 2]
    assert np.allclose(distances[0], np.asarray([0.1, 0.2], dtype=np.float32))
