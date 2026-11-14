# RB-CFG-EDGE: Configuration error on access or cell-site node
- fault_type: config_error
- node_types: edge
- action: rollback_config
- risk: low

## Symptoms
A configuration error on a access cell edge tier node shows as sudden change after a change window, routing or mtu mismatch, session flaps. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check the change log for the last two hours and diff the running configuration against the golden copy. Confirm the alarmed node sits at the access cell edge tier and count its downstream dependents before acting.

## Remediation
Roll back to the last known good configuration from the change record and verify sessions and routes re-establish. Applies to configuration error at the access cell edge tier.

## Risk
Risk level low. A mistaken action at the access cell edge tier affects few downstream nodes, so review the blast radius before execution.
