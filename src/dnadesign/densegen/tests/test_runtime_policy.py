# ABOUTME: Tests runtime policy stall and warning timing behavior.
# ABOUTME: Ensures stall timers reset on solver progress signals.

from __future__ import annotations

from dnadesign.densegen.src.core.runtime_policy import RuntimePolicy


def test_stall_timer_resets_on_progress() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        arrays_generated_before_resample=1,
        stall_seconds_before_resample=10,
        stall_warning_every_seconds=5,
        max_consecutive_failures=25,
        max_seconds_per_plan=0,
    )

    assert policy.should_trigger_stall(now=9.0, last_progress=0.0) is False
    assert policy.should_trigger_stall(now=10.0, last_progress=0.0) is True

    assert policy.should_warn_stall(now=4.0, last_warn=0.0, last_progress=0.0) is False
    assert policy.should_warn_stall(now=5.0, last_warn=0.0, last_progress=0.0) is True

    last_progress = 6.0
    last_warn = 5.0
    assert policy.should_warn_stall(now=9.0, last_warn=last_warn, last_progress=last_progress) is False
    assert policy.should_warn_stall(now=11.0, last_warn=last_warn, last_progress=last_progress) is True
