## Retron Command Surfaces

Command fragments group Retron preflight commands by owner lane.

- `compiler.yaml`: Retron MSD lint and compile commands.
- `materialize.yaml`: Retron MSD single-unit sequence-bundle materialization
  command.
- `snapback.yaml`: released-product Snapback probe command.
- `yiu.yaml`: YIU contrast validation command.

The scar-nick regeneration lane is a runtime command group at
`docs/studies/retron_hairpin_design/operations/runtime/command-groups/lanes/scar-nick.yaml`;
do not duplicate it here unless it becomes a readiness contract check.
