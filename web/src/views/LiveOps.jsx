import React, { useMemo, useState } from 'react'
import { api } from '../api.js'
import { useApi } from '../hooks.js'
import { Async } from '../components/ui.jsx'
import Topology from '../components/Topology.jsx'

const FAULTS = ['fibre_cut', 'power_failure', 'congestion', 'hardware_degradation', 'config_error']
const SEV = { critical: 'text-red-400', major: 'text-amber-400', minor: 'text-sky-300' }

export default function LiveOps({ openIncident }) {
  const topo = useApi(api.topology, [])
  const alarms = useApi(() => api.alarms(100), [], 3000)
  const incidents = useApi(api.incidents, [], 3000)
  const [showNoise, setShowNoise] = useState(true)
  const [fault, setFault] = useState('fibre_cut')
  const [node, setNode] = useState('')
  const [msg, setMsg] = useState(null)
  const [busy, setBusy] = useState(false)

  const incList = incidents.data?.incidents || []
  const open = incList.find((i) => i.status === 'open') || incList[0]
  const rootId = open?.rca?.root_candidates?.[0]?.node_id

  const nodeState = useMemo(() => {
    const s = {}
    for (const a of alarms.data?.alarms || []) {
      if (a.is_valid_pred === false) continue
      const cur = s[a.node_id]
      const next = a.severity === 'critical' ? 'alarm' : 'warn'
      if (cur !== 'alarm') s[a.node_id] = next
    }
    return s
  }, [alarms.data])

  async function inject() {
    setBusy(true); setMsg(null)
    try {
      const r = await api.inject(fault, node)
      setMsg(`Injected ${r.fault_type} on ${r.node_id} (${r.incident_id})`)
      alarms.reload(); incidents.reload()
    } catch (e) { setMsg(`Inject failed: ${e.message}`) } finally { setBusy(false) }
  }

  return (
    <div className="grid lg:grid-cols-3 gap-3">
      <section className="panel lg:col-span-2" aria-label="Topology">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <h2 className="h mb-0">Topology {rootId && <span className="text-sky-300 normal-case">root candidate: {rootId}</span>}</h2>
          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="fault">Fault type</label>
            <select id="fault" className="bg-slate-800 border border-slate-600 rounded px-2 py-1" value={fault} onChange={(e) => setFault(e.target.value)}>
              {FAULTS.map((f) => <option key={f}>{f}</option>)}
            </select>
            <label className="sr-only" htmlFor="node">Target node</label>
            <input id="node" placeholder="node (optional)" className="bg-slate-800 border border-slate-600 rounded px-2 py-1 w-32" value={node} onChange={(e) => setNode(e.target.value)} />
            <button className="btn" onClick={inject} disabled={busy}>Inject fault</button>
          </div>
        </div>
        {msg && <div role="status" className="text-xs text-emerald-300 mb-2">{msg}</div>}
        <Async state={topo} empty={(d) => !d.nodes?.length}>
          {(t) => <Topology topology={t} nodeState={nodeState} rootId={rootId} onSelect={setNode} />}
        </Async>
        <div className="flex gap-4 text-xs text-slate-400 mt-2">
          <span><i className="inline-block w-2 h-2 rounded-full bg-emerald-500" /> ok</span>
          <span><i className="inline-block w-2 h-2 rounded-full bg-amber-500" /> warn</span>
          <span><i className="inline-block w-2 h-2 rounded-full bg-red-500" /> critical</span>
          <span><i className="inline-block w-2 h-2 rounded-full border-2 border-sky-400" /> root cause</span>
        </div>
      </section>

      <div className="space-y-3">
        <section className="panel" aria-label="Incidents">
          <h2 className="h">Incidents</h2>
          <Async state={incidents} empty={(d) => !d.incidents?.length}>
            {(d) => (
              <ul>
                {d.incidents.map((i) => (
                  <li key={i.incident_id}>
                    <button className="w-full text-left hover:bg-slate-800 rounded px-1 py-1" onClick={() => openIncident(i.incident_id)}>
                      <b>{i.incident_id}</b> <span className="text-slate-400">{i.status}</span>{' '}
                      {i.rca?.fault_type?.label} @ {i.rca?.root_candidates?.[0]?.node_id}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Async>
        </section>
        <section className="panel" aria-label="Alarm feed">
          <div className="flex items-center justify-between">
            <h2 className="h mb-0">Alarm feed (3s)</h2>
            <label className="text-xs flex items-center gap-1">
              <input type="checkbox" checked={showNoise} onChange={(e) => setShowNoise(e.target.checked)} /> show noise
            </label>
          </div>
          <Async state={alarms} empty={(d) => !d.alarms?.length}>
            {(d) => (
              <ul className="mt-2 max-h-72 overflow-auto space-y-1" data-testid="alarm-feed">
                {d.alarms.filter((a) => showNoise || a.is_valid_pred !== false).map((a) => (
                  <li key={a.alarm_id} data-noise={a.is_valid_pred === false} className={a.is_valid_pred === false ? 'opacity-40' : ''}>
                    <span className="text-slate-500">{a.ts.slice(11, 19)}</span>{' '}
                    <span className={SEV[a.severity]}>{a.severity}</span> {a.node_id} {a.alarm_type}
                    {a.is_valid_pred === false && <span className="ml-1 text-slate-400">[noise {a.valid_prob?.toFixed(2)}]</span>}
                  </li>
                ))}
              </ul>
            )}
          </Async>
        </section>
      </div>
    </div>
  )
}
