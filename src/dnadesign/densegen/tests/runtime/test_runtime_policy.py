# ABOUTME: Tests runtime policy stall and warning timing behavior.
# ABOUTME: Ensures stall timers reset on solver progress signals.

from __future__ import annotations

from dnadesign.densegen.src.core.runtime_policy import RuntimePolicy


def test_stall_timer_resets_on_progress() -> None:
    policy = RuntimePolicy(
        pool_strategy="subsample",
        max_accepted_per_library=1,
        no_progress_seconds_before_resample=10,
        max_consecutive_no_progress_resamples=25,
    )

    assert policy.should_trigger_stall(now=9.0, last_progress=0.0) is False
    assert policy.should_trigger_stall(now=10.0, last_progress=0.0) is True

    assert policy.should_warn_stall(now=9.0, last_warn=0.0, last_progress=0.0) is False
    assert policy.should_warn_stall(now=10.0, last_warn=0.0, last_progress=0.0) is True

    last_progress = 6.0
    last_warn = 10.0
    assert policy.should_warn_stall(now=15.0, last_warn=last_warn, last_progress=last_progress) is False
    assert policy.should_warn_stall(now=16.0, last_warn=last_warn, last_progress=last_progress) is False
    assert policy.should_warn_stall(now=20.0, last_warn=last_warn, last_progress=last_progress) is True
