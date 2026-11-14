# RB-CONG-AGG: Traffic congestion on aggregation node
- fault_type: congestion
- node_types: agg
- action: reroute_traffic
- risk: medium

## Symptoms
A congestion on a aggregation tier node shows as latency and packet loss rising together, throughput flat at link capacity. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check link utilisation and top talkers; look for a recent event, release or traffic shift that explains the load. Confirm the alarmed node sits at the aggregation tier and count its downstream dependents before acting.

## Remediation
Shift traffic to the backup path or a sibling node using the pre-approved routing policy, then confirm packet loss falls below 1 percent. Applies to congestion at the aggregation tier.

## Risk
Risk level medium. A mistaken action at the aggregation tier affects some downstream nodes, so a human must approve before execution.
