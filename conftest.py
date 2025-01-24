import pytest
import torch

@pytest.fixture
def sample_sequence_data():
    """Fixture to provide sample sequence data."""
    return [
        {"id": "seq1", "sequence": "ACTG...", "meta_type": "promoter"},
        {"id": "seq2", "sequence": "GATC...", "meta_type": "tfbs"},
    ]

@pytest.fixture
def sample_pt_file(tmp_path, sample_sequence_data):
    """Fixture to create a temporary .pt file with sample data."""
    file_path = tmp_path / "sample_sequences.pt"
    torch.save(sample_sequence_data, file_path)
    return file_path
