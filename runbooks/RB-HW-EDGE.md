# RB-HW-EDGE: Hardware degradation on access or cell-site node
- fault_type: hardware_degradation
- node_types: edge
- action: restart_node
- risk: low

## Symptoms
A hardware degradation on a access cell edge tier node shows as slow rise in temperature and cpu, intermittent errors, fan or line-card warnings. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check temperature, fan and line-card counters over the last 24 hours; compare with peer nodes of the same vendor. Confirm the alarmed node sits at the access cell edge tier and count its downstream dependents before acting.

## Remediation
Restart the node in a controlled way during low traffic, then verify that cpu, memory and temperature return to baseline. Applies to hardware degradation at the access cell edge tier.

## Risk
Risk level low. A mistaken action at the access cell edge tier affects few downstream nodes, so review the blast radius before execution.
