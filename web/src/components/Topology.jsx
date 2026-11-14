import React, { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'

const STATE_COLOR = { ok: '#10b981', warn: '#f59e0b', alarm: '#ef4444' }

/** nodeState: {node_id: 'ok'|'warn'|'alarm'}, rootId: ringed node. */
export default function Topology({ topology, nodeState, rootId, onSelect }) {
  const ref = useRef(null)
  const cyRef = useRef(null)

  useEffect(() => {
    const cy = cytoscape({
      container: ref.current,
      elements: [
        ...topology.nodes.map((n) => ({ data: { id: n.node_id, label: n.node_id, type: n.node_type } })),
        ...topology.links.map((l) => ({ data: { id: l.link_id, source: l.src, target: l.dst } })),
      ],
      style: [
        { selector: 'node', style: { label: 'data(label)', color: '#cbd5e1', 'font-size': 9, 'text-valign': 'bottom', 'text-margin-y': 3, 'background-color': STATE_COLOR.ok, width: 18, height: 18, 'border-width': 0 } },
        { selector: 'node[type="core"]', style: { shape: 'diamond', width: 26, height: 26 } },
        { selector: 'node[type="agg"]', style: { shape: 'round-rectangle', width: 22, height: 22 } },
        { selector: 'edge', style: { width: 1.5, 'line-color': '#475569' } },
        { selector: 'node.root', style: { 'border-width': 4, 'border-color': '#38bdf8' } },
      ],
      layout: { name: 'breadthfirst', directed: true, padding: 12, spacingFactor: 1.1 },
      userZoomingEnabled: true,
    })
    cy.on('tap', 'node', (e) => onSelect && onSelect(e.target.id()))
    cyRef.current = cy
    return () => cy.destroy()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topology])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().forEach((n) => {
      n.style('background-color', STATE_COLOR[nodeState[n.id()] || 'ok'])
      n.toggleClass('root', n.id() === rootId)
    })
  }, [nodeState, rootId, topology])

  return <div ref={ref} data-testid="topology" role="img" aria-label="Network topology" className="w-full h-[420px] bg-slate-950 rounded border border-slate-800" />
}
