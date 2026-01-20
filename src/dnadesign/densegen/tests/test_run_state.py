from __future__ import annotations

from dnadesign.densegen.src.core.run_paths import ensure_run_meta_dir, run_state_path
from dnadesign.densegen.src.core.run_state import RunState, load_run_state


def test_run_state_roundtrip(tmp_path) -> None:
    counts = {("input", "plan"): 3, ("input", "plan2"): 1}
    state = RunState.from_counts(
        run_id="demo",
        schema_version="2.4",
        config_sha256="abc123",
        run_root="/tmp/demo",
        counts=counts,
        created_at="2026-01-18T00:00:00Z",
        updated_at="2026-01-18T00:00:00Z",
    )
    ensure_run_meta_dir(tmp_path)
    path = run_state_path(tmp_path)
    state.write_json(path)
    loaded = load_run_state(path)
    assert loaded.run_id == "demo"
    assert loaded.schema_version == "2.4"
    assert loaded.items[0].generated >= 0
