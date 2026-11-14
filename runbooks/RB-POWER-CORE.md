# RB-POWER-CORE: Power failure on core router
- fault_type: power_failure
- node_types: core
- action: reroute_traffic
- risk: high

## Symptoms
A power failure on a core tier node shows as node unreachable, sudden loss of all KPIs, battery or rectifier alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check mains and battery status on the site monitor; ask whether neighbouring sites in the same feeder are also dark. Confirm the alarmed node sits at the core tier and count its downstream dependents before acting.

## Remediation
Shift traffic to the backup path or a sibling node using the pre-approved routing policy, then confirm packet loss falls below 1 percent. Applies to power failure at the core tier.

## Risk
Risk level high. A mistaken action at the core tier affects many downstream nodes, so a human must approve before execution.
