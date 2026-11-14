# RB-FIBRE-AGG: Fibre cut on aggregation node
- fault_type: fibre_cut
- node_types: agg
- action: reroute_traffic
- risk: medium

## Symptoms
A fibre cut on a aggregation tier node shows as loss of signal, packet loss near 100 percent on the uplink, optical power alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check loss of signal and optical receive power on both ends of the link; compare with the last maintenance record. Confirm the alarmed node sits at the aggregation tier and count its downstream dependents before acting.

## Remediation
Shift traffic to the backup path or a sibling node using the pre-approved routing policy, then confirm packet loss falls below 1 percent. Applies to fibre cut at the aggregation tier.

## Risk
Risk level medium. A mistaken action at the aggregation tier affects some downstream nodes, so a human must approve before execution.
