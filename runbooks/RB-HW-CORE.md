# RB-HW-CORE: Hardware degradation on core router
- fault_type: hardware_degradation
- node_types: core
- action: reroute_traffic
- risk: high

## Symptoms
A hardware degradation on a core tier node shows as slow rise in temperature and cpu, intermittent errors, fan or line-card warnings. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check temperature, fan and line-card counters over the last 24 hours; compare with peer nodes of the same vendor. Confirm the alarmed node sits at the core tier and count its downstream dependents before acting.

## Remediation
Shift traffic to the backup path or a sibling node using the pre-approved routing policy, then confirm packet loss falls below 1 percent. Applies to hardware degradation at the core tier.

## Risk
Risk level high. A mistaken action at the core tier affects many downstream nodes, so a human must approve before execution.
