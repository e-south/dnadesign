"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/test_parser.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import tempfile
from dnadesign.densehairpins.parser import parse_meme_file

def test_parse_meme_file_consensus_only():
    # Create a temporary file simulating a MEME file with a "Multilevel" line.
    content = """Header info here
Multilevel   ATGCGTACGTA
Additional info
"""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = parse_meme_file(tmp_path, consensus_only=True)
        assert result == "ATGCGTACGTA", f"Expected 'ATGCGTACGTA' but got {result}"
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_parse_meme_file_consensus_only()
    print("test_parse_meme_file_consensus_only passed.")
