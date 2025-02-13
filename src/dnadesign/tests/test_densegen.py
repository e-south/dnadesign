"""
--------------------------------------------------------------------------------
<dnadesign project>

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pytest
import pandas as pd
import torch
from pathlib import Path
import tempfile
import datetime
import time

from dnadesign.densegen.sampler import TFSampler
from dnadesign.utils import generate_sequence_entry, SequenceSaver
from dnadesign.densegen.progress_tracker import ProgressTracker
from dnadesign.densegen.optimizer_wrapper import DenseArrayOptimizer, random_fill

# --- Dummy Classes for Testing the Optimizer and Gap-Fill Logic ---

class DummyDenseArray:
    """
    A dummy implementation of a DenseArray solution.
    """
    def __init__(self, sequence, offsets, nb_motifs):
        self.sequence = sequence  # The raw sequence (e.g., "ACGT")
        self._offsets = offsets   # A list of dummy offset tuples (for testing)
        self.nb_motifs = nb_motifs
        # For simplicity, let compression_ratio be a fixed value.
        self.compression_ratio = 1.0
        self.meta_gap_fill = False
        self.meta_gap_fill_details = None

    def offset_indices_in_order(self):
        # For testing, simply return the provided offsets.
        return self._offsets

    def __str__(self):
        # Return a dummy visual representation.
        return f"DummyVisual: {self.sequence}"

class DummyOptimizer:
    """
    A dummy optimizer that always returns the given dummy solution.
    Also implements a minimal dummy model with a SetTimeLimit method.
    """
    def __init__(self, dummy_solution):
        self.dummy_solution = dummy_solution

    def optimal(self, solver, solver_options):
        return self.dummy_solution

    def build_model(self, solver, solver_options):
        # Create a dummy model object.
        class DummyModel:
            def __init__(self):
                self.time_limit = None
            def SetTimeLimit(self, ms):
                self.time_limit = ms
            def SupportsTimeLimit(self):
                return True
        self.model = DummyModel()

    def solve(self):
        return self.dummy_solution

    def forbid(self, solution):
        # Dummy forbid does nothing.
        pass

# Subclass DenseArrayOptimizer to override get_optimizer_instance to return our dummy optimizer.
class DummyDenseArrayOptimizer(DenseArrayOptimizer):
    def __init__(self, dummy_solution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_solution = dummy_solution

    def get_optimizer_instance(self):
        return DummyOptimizer(self.dummy_solution)

# --- Tests ---

def test_tfsampler_sampling():
    """
    Test that TFSampler properly samples unique TFs and then a corresponding binding site for each.
    """
    data = {
        "tf": ["TF1", "TF1", "TF2", "TF3", "TF3", "TF4"],
        "tfbs": ["bs1", "bs2", "bs3", "bs4", "bs5", "bs6"],
        "deg_source": ["source1"] * 6
    }
    df = pd.DataFrame(data)
    sampler = TFSampler(df)
    # Request 3 unique TFs.
    result = sampler.subsample_binding_sites(sample_size=3, unique_tf_only=True)
    tfs = [r[0] for r in result]
    # Assert that we have exactly 3 entries with unique TF values.
    assert len(result) == 3
    assert len(set(tfs)) == 3
    # Optionally, check that each returned binding site is one of the binding sites associated with its TF.
    for tf, tfbs, _ in result:
        tf_rows = df[df['tf'] == tf]
        assert tfbs in tf_rows['tfbs'].values

def test_generate_sequence_entry():
    """
    Test that generate_sequence_entry returns a dictionary with the expected keys and non-null values.
    """
    # Create a dummy solution with a known sequence and offsets.
    dummy_solution = DummyDenseArray("ACGT", [(0, 0), (2, 1)], 2)
    config = {
        "sequence_length": 100,
        "quota": 5,
        "subsample_size": 10,
        "arrays_generated_before_resample": 1,
        "solver": "CBC",
        "solver_options": ["Threads=16", "TimeLimit=20"],
        "fixed_elements": {"promoter_constraints": []}
    }
    entry = generate_sequence_entry(dummy_solution, ["TFBS_SRC"], ["TF1_bs1"], config)
    for key in ["id", "meta_date_accessed", "meta_source", "meta_sequence_visual", "sequence", "config"]:
        assert key in entry, f"Key '{key}' not found in generated entry."
        assert entry[key] is not None, f"Key '{key}' is None."

def test_save_and_load_pt(tmp_path):
    """
    Test that SequenceSaver saves a list of dictionaries to a .pt file, and that file loads correctly.
    """
    entries = [
        {"id": "test1", "sequence": "ACGT", "meta_source": "dummy_source"},
        {"id": "test2", "sequence": "TGCA", "meta_source": "dummy_source"}
    ]
    saver = SequenceSaver(str(tmp_path))
    filename = "test.pt"
    saver.save(entries, filename)
    loaded = torch.load(tmp_path / filename)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    for entry in loaded:
        for key in ["id", "sequence", "meta_source"]:
            assert key in entry and entry[key], f"Key '{key}' missing or empty in entry: {entry}"

def test_gap_fill():
    """
    Test that when gap fill is enabled with a 5prime fill, the final solution sequence
    is extended to the desired sequence_length and that the original sequence appears at the end.
    """
    # Create a dummy solution with a short sequence.
    dummy_solution = DummyDenseArray("ACGT", [(0, 0)], 1)
    # Set target sequence length greater than the current sequence.
    target_length = 10
    # Create a DummyDenseArrayOptimizer that will always return our dummy_solution.
    optimizer = DummyDenseArrayOptimizer(
        dummy_solution,
        library=["motif1", "motif2"],
        sequence_length=target_length,
        solver="CBC",
        solver_options=["TimeLimit=5"],
        fixed_elements={},
        fill_gap=True,
        fill_gap_end="5prime",
        fill_gc_min=0.40,
        fill_gc_max=0.60
    )
    # Override the dummy solution's sequence to be short.
    dummy_solution.sequence = "ACGT"
    # Run optimize() which (if gap fill is triggered) should update the solution.
    solution = optimizer.optimize(timeout_seconds=10)
    # Assert that the solution.sequence length is target_length.
    assert len(solution.sequence) == target_length, "Gap fill did not extend the sequence to the target length."
    # Check that gap fill flag is set.
    assert getattr(solution, "meta_gap_fill", False) is True, "meta_gap_fill attribute not set after gap fill."
    # Since we are filling at the 5prime end, the original "ACGT" should appear at the end.
    assert solution.sequence.endswith("ACGT"), "Original sequence not found at the end after 5prime gap fill."

if __name__ == "__main__":
    pytest.main()
