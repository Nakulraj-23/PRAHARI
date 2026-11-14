# RB-FIBRE-EDGE: Fibre cut on access or cell-site node
- fault_type: fibre_cut
- node_types: edge
- action: dispatch_field_team
- risk: low

## Symptoms
A fibre cut on a access cell edge tier node shows as loss of signal, packet loss near 100 percent on the uplink, optical power alarms. Downstream nodes raise dependent alarms within a few minutes; treat these as symptoms and not as separate faults.

## Checks
Check loss of signal and optical receive power on both ends of the link; compare with the last maintenance record. Confirm the alarmed node sits at the access cell edge tier and count its downstream dependents before acting.

## Remediation
Raise a field ticket with the node id, site and suspected fault, and keep traffic on the backup path until the crew confirms the repair. Applies to fibre cut at the access cell edge tier.

## Risk
Risk level low. A mistaken action at the access cell edge tier affects few downstream nodes, so review the blast radius before execution.
