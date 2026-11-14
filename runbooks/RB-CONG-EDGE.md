# RB-CONG-EDGE: Traffic congestion on access or cell-site node
- fault_type: congestion
- node_types: edge
- action: throttle_traffic
- risk: low

## Symptoms
A congestion on a access cell edge tier node shows as latency and packet loss rising together, throughput flat at link capacity. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check link utilisation and top talkers; look for a recent event, release or traffic shift that explains the load. Confirm the alarmed node sits at the access cell edge tier and count its downstream dependents before acting.

## Remediation
Apply rate limits to the lowest-priority traffic classes on the congested links and re-check utilisation after five minutes. Applies to congestion at the access cell edge tier.

## Risk
Risk level low. A mistaken action at the access cell edge tier affects few downstream nodes, so review the blast radius before execution.
