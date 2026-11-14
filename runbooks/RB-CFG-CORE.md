# RB-CFG-CORE: Configuration error on core router
- fault_type: config_error
- node_types: core
- action: rollback_config
- risk: high

## Symptoms
A configuration error on a core tier node shows as sudden change after a change window, routing or mtu mismatch, session flaps. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check the change log for the last two hours and diff the running configuration against the golden copy. Confirm the alarmed node sits at the core tier and count its downstream dependents before acting.

## Remediation
Roll back to the last known good configuration from the change record and verify sessions and routes re-establish. Applies to configuration error at the core tier.

## Risk
Risk level high. A mistaken action at the core tier affects many downstream nodes, so a human must approve before execution.
