# RB-CFG-AGG: Configuration error on aggregation node
- fault_type: config_error
- node_types: agg
- action: rollback_config
- risk: medium

## Symptoms
A configuration error on a aggregation tier node shows as sudden change after a change window, routing or mtu mismatch, session flaps. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check the change log for the last two hours and diff the running configuration against the golden copy. Confirm the alarmed node sits at the aggregation tier and count its downstream dependents before acting.

## Remediation
Roll back to the last known good configuration from the change record and verify sessions and routes re-establish. Applies to configuration error at the aggregation tier.

## Risk
Risk level medium. A mistaken action at the aggregation tier affects some downstream nodes, so a human must approve before execution.
