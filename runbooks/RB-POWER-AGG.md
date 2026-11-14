# RB-POWER-AGG: Power failure on aggregation node
- fault_type: power_failure
- node_types: agg
- action: dispatch_field_team
- risk: medium

## Symptoms
A power failure on a aggregation tier node shows as node unreachable, sudden loss of all KPIs, battery or rectifier alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check mains and battery status on the site monitor; ask whether neighbouring sites in the same feeder are also dark. Confirm the alarmed node sits at the aggregation tier and count its downstream dependents before acting.

## Remediation
Raise a field ticket with the node id, site and suspected fault, and keep traffic on the backup path until the crew confirms the repair. Applies to power failure at the aggregation tier.

## Risk
Risk level medium. A mistaken action at the aggregation tier affects some downstream nodes, so a human must approve before execution.
