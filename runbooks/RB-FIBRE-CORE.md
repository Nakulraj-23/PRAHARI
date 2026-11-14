# RB-FIBRE-CORE: Fibre cut on core router
- fault_type: fibre_cut
- node_types: core
- action: reroute_traffic
- risk: high

## Symptoms
A fibre cut on a core tier node shows as loss of signal, packet loss near 100 percent on the uplink, optical power alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check loss of signal and optical receive power on both ends of the link; compare with the last maintenance record. Confirm the alarmed node sits at the core tier and count its downstream dependents before acting.

## Remediation
Shift traffic to the backup path or a sibling node using the pre-approved routing policy, then confirm packet loss falls below 1 percent. Applies to fibre cut at the core tier.

## Risk
Risk level high. A mistaken action at the core tier affects many downstream nodes, so a human must approve before execution.
