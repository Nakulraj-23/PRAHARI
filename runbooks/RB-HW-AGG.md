# RB-HW-AGG: Hardware degradation on aggregation node
- fault_type: hardware_degradation
- node_types: agg
- action: dispatch_field_team
- risk: medium

## Symptoms
A hardware degradation on a aggregation tier node shows as slow rise in temperature and cpu, intermittent errors, fan or line-card warnings. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check temperature, fan and line-card counters over the last 24 hours; compare with peer nodes of the same vendor. Confirm the alarmed node sits at the aggregation tier and count its downstream dependents before acting.

## Remediation
Raise a field ticket with the node id, site and suspected fault, and keep traffic on the backup path until the crew confirms the repair. Applies to hardware degradation at the aggregation tier.

## Risk
Risk level medium. A mistaken action at the aggregation tier affects some downstream nodes, so a human must approve before execution.
