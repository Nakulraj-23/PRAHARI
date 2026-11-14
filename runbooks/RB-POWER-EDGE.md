# RB-POWER-EDGE: Power failure on access or cell-site node
- fault_type: power_failure
- node_types: edge
- action: dispatch_field_team
- risk: low

## Symptoms
A power failure on a access cell edge tier node shows as node unreachable, sudden loss of all KPIs, battery or rectifier alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check mains and battery status on the site monitor; ask whether neighbouring sites in the same feeder are also dark. Confirm the alarmed node sits at the access cell edge tier and count its downstream dependents before acting.

## Remediation
Raise a field ticket with the node id, site and suspected fault, and keep traffic on the backup path until the crew confirms the repair. Applies to power failure at the access cell edge tier.

## Risk
Risk level low. A mistaken action at the access cell edge tier affects few downstream nodes, so review the blast radius before execution.
