"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/__init__.py

Profile and workspace configuration helpers for notify setup and watch flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .flows import (
    SetupEventsResolution,
    create_wizard_profile,
    resolve_profile_path_for_setup,
    resolve_profile_path_for_wizard,
    resolve_setup_events,
    resolve_webhook_config,
)
from .ops import sanitize_profile_name, wizard_next_steps
from .policy import (
    DEFAULT_PROFILE_PATH,
    DEFAULT_WEBHOOK_ENV,
    default_profile_path_for_tool,
    normalize_policy_name,
    policy_defaults,
    register_workflow_policy,
    resolve_workflow_policy,
    supported_workflow_policies,
)
from .schema import (
    PROFILE_ALLOWED_KEYS,
    PROFILE_REQUIRED_KEYS,
    PROFILE_VERSION,
    WEBHOOK_SOURCES,
    read_profile,
    resolve_profile_events_source,
    resolve_profile_webhook_source,
    validate_events_source_config,
    validate_webhook_config,
)
from .workspace import (
    ToolWorkspaceResolver,
    list_tool_workspaces,
    normalize_tool_name,
    register_tool_workspace_resolver,
    resolve_tool_workspace_config_path,
)

__all__ = [
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_WEBHOOK_ENV",
    "PROFILE_ALLOWED_KEYS",
    "PROFILE_REQUIRED_KEYS",
    "PROFILE_VERSION",
    "SetupEventsResolution",
    "ToolWorkspaceResolver",
    "WEBHOOK_SOURCES",
    "create_wizard_profile",
    "default_profile_path_for_tool",
    "list_tool_workspaces",
    "normalize_policy_name",
    "normalize_tool_name",
    "policy_defaults",
    "read_profile",
    "register_tool_workspace_resolver",
    "register_workflow_policy",
    "resolve_profile_events_source",
    "resolve_profile_webhook_source",
    "resolve_profile_path_for_setup",
    "resolve_profile_path_for_wizard",
    "resolve_setup_events",
    "resolve_tool_workspace_config_path",
    "resolve_webhook_config",
    "resolve_workflow_policy",
    "sanitize_profile_name",
    "supported_workflow_policies",
    "validate_events_source_config",
    "validate_webhook_config",
    "wizard_next_steps",
]
