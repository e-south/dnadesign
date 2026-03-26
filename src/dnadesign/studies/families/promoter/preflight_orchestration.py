"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/preflight_orchestration.py

Study-owned preflight builders for orchestration and notify environment
surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from dnadesign.ops.preflight import EnvironmentCheckTarget, PreflightCheck, build_environment_checks

_NOTIFY_WEBHOOK_ENV_KEYS = ("NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE")
_NOTIFY_TLS_ENV_KEY = "SSL_CERT_FILE"


def resolve_notify_environment_state(
    *,
    environ: Mapping[str, object | None],
) -> dict[str, bool]:
    return {
        env_var: bool(str(environ.get(env_var) or "").strip())
        for env_var in (*_NOTIFY_WEBHOOK_ENV_KEYS, _NOTIFY_TLS_ENV_KEY)
    }


def build_promoter_preflight_notify_environment_checks(
    *,
    notify_env_state: Mapping[str, bool],
    notify_environment_phase_id: str,
    enabled_groups: Collection[str],
) -> tuple[PreflightCheck, ...]:
    if "notify_environment" not in enabled_groups:
        return ()
    return build_environment_checks(
        flag_state=notify_env_state,
        targets=(
            EnvironmentCheckTarget(
                check_id="notify.environment.webhook",
                check_group="notify_environment",
                phase="notify",
                phase_id=notify_environment_phase_id,
                flag_names=_NOTIFY_WEBHOOK_ENV_KEYS,
                match_mode="any",
                ok_summary="batch notify secret is configured in the environment",
                missing_summary=("batch notify secret is not configured; export NOTIFY_WEBHOOK_FILE or NOTIFY_WEBHOOK"),
            ),
            EnvironmentCheckTarget(
                check_id="notify.environment.tls",
                check_group="notify_environment",
                phase="notify",
                phase_id=notify_environment_phase_id,
                flag_names=(_NOTIFY_TLS_ENV_KEY,),
                match_mode="all",
                ok_summary="SSL_CERT_FILE is configured for notify profile doctor and live delivery",
                missing_summary="SSL_CERT_FILE is not configured for notify profile doctor and live delivery",
            ),
        ),
    )


__all__ = [
    "build_promoter_preflight_notify_environment_checks",
    "resolve_notify_environment_state",
]
