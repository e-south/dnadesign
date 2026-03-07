## Notify maintainer architecture map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-01

This page maps package boundaries for maintainers extending command registration, event resolution, and delivery/runtime flows.

### Command surface

- `src/dnadesign/notify/cli/__init__.py`: Typer app router and command-group registration.
- `src/dnadesign/notify/cli/bindings/__init__.py`: binding surface used by handlers and tests.
- `src/dnadesign/notify/cli/bindings/deps/`: dependency exports by domain (`profile`, `setup`, `runtime`, `send`).
- `src/dnadesign/notify/cli/bindings/registry.py`: command registration wiring.
- `src/dnadesign/notify/cli/commands/`: option declarations and command registration modules.
- `src/dnadesign/notify/cli/handlers/`: command execution handlers.

### Event-source and profile boundaries

- `src/dnadesign/notify/events/source.py`: tool resolver registry.
- `src/dnadesign/notify/events/source_builtin.py`: built-in tool resolvers (`densegen`, `infer`).
- `src/dnadesign/notify/profiles/flow_events.py`: setup event-source mode resolution.
- `src/dnadesign/notify/profiles/flow_webhook.py`: webhook source selection and secure ref storage.
- `src/dnadesign/notify/profiles/flow_profile.py`: profile materialization and default path resolution.
- `src/dnadesign/notify/profiles/workspace.py`: workspace-name to config-path resolver.
- `src/dnadesign/notify/profiles/schema/`: profile schema contract and loader validation.

### Tool-event transforms and policy logic

- `src/dnadesign/notify/events/transforms.py`: USR event to Notify status/message/meta transforms.
- `src/dnadesign/notify/tool_events/core.py`: tool-event evaluator registry.
- `src/dnadesign/notify/tool_events/packs_builtin.py`: built-in tool-event pack installers.
- `src/dnadesign/notify/tool_events/densegen*.py`: DenseGen-specific metric extraction, message rendering, and emission state.

### Delivery and runtime loop boundaries

- `src/dnadesign/notify/runtime/watch_runner.py`: watch orchestration entrypoint.
- `src/dnadesign/notify/runtime/watch_runner_contract.py`: watch command contract validation/coercion.
- `src/dnadesign/notify/runtime/watch_runner_resolution.py`: profile, webhook, and events-source resolution.
- `src/dnadesign/notify/runtime/watch_events.py`: event parse/filter/payload preparation.
- `src/dnadesign/notify/runtime/watch_delivery.py`: dry-run output and live delivery/spool outcomes.
- `src/dnadesign/notify/runtime/spool_runner.py`: spool replay orchestration.
- `src/dnadesign/notify/runtime/cursor/`: cursor offset, lock, and follow-loop primitives.
- `src/dnadesign/notify/delivery/http.py`: webhook HTTP posting.
- `src/dnadesign/notify/delivery/secrets/`: secret backend contracts and backend operations.
- `src/dnadesign/notify/providers/`: provider payload adapters (`generic`, `slack`, `discord`).

### Import and extension contract

- Runtime code and tests should import from canonical subpackages (`delivery`, `events`, `profiles`, `runtime`, `tool_events`).
- Command wiring changes should route through CLI binding registries, not ad-hoc command registration.
- New tool integrations should define resolver behavior and event-pack mapping before runtime hooks are added.
