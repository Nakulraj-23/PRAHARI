# RB-CONG-CORE: Traffic congestion on core router
- fault_type: congestion
- node_types: core
- action: throttle_traffic
- risk: high

## Symptoms
A congestion on a core tier node shows as latency and packet loss rising together, throughput flat at link capacity. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check link utilisation and top talkers; look for a recent event, release or traffic shift that explains the load. Confirm the alarmed node sits at the core tier and count its downstream dependents before acting.

## Remediation
Apply rate limits to the lowest-priority traffic classes on the congested links and re-check utilisation after five minutes. Applies to congestion at the core tier.

## Risk
Risk level high. A mistaken action at the core tier affects many downstream nodes, so a human must approve before execution.
