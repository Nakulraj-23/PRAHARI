import React, { useState } from 'react'
import { BarChart, Bar as RBar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { api } from '../api.js'
import { useApi } from '../hooks.js'
import { Async, Bar, STATUS_COLOR } from '../components/ui.jsx'

export default function Incident({ incidentId, setIncidentId }) {
  const list = useApi(api.incidents, [])
  const id = incidentId || list.data?.incidents?.[0]?.incident_id
  const detail = useApi(() => (id ? api.incident(id) : Promise.resolve(null)), [id])
  const [lang, setLang] = useState('en')
  const explain = useApi(() => (id ? api.explain(id, lang) : Promise.resolve(null)), [id, lang])
  const [note, setNote] = useState('')
  const [user, setUser] = useState('noc.operator')
  const [result, setResult] = useState(null)

  async function decide(verb, actionId) {
    setResult(null)
    try {
      await api.decide(actionId, verb, user, note)
      setResult(`Action ${actionId} ${verb === 'approve' ? 'approved' : 'rejected'}`)
      detail.reload()
    } catch (e) { setResult(`Failed: ${e.message}`) }
  }

  return (
    <div className="space-y-3">
      <Async state={list} empty={(d) => !d.incidents?.length}>
        {(d) => (
          <div className="flex items-center gap-2">
            <label htmlFor="inc">Incident</label>
            <select id="inc" className="bg-slate-800 border border-slate-600 rounded px-2 py-1" value={id} onChange={(e) => setIncidentId(e.target.value)}>
              {d.incidents.map((i) => <option key={i.incident_id}>{i.incident_id}</option>)}
            </select>
          </div>
        )}
      </Async>
      {id && (
        <Async state={detail}>
          {(d) => {
            if (!d) return null
            const rca = d.rca
            const top = rca.root_candidates.slice(0, 3)
            const rootId = top[0]?.node_id
            const deltas = rca.evidence.kpi_deltas?.[rootId] || {}
            const chart = Object.entries(deltas).map(([k, v]) => ({ kpi: k, delta: v }))
            const gov = d.action?.governance || d.governance
            const status = d.action?.status
            return (
              <div className="grid lg:grid-cols-2 gap-3">
                <section className="panel" aria-label="Root cause">
                  <h2 className="h">Root candidates</h2>
                  <ul className="space-y-1">
                    {top.map((c) => (
                      <li key={c.node_id}><div className="flex gap-2"><span className="w-20">{c.node_id}</span><div className="flex-1"><Bar value={c.prob} label={`${c.node_id} probability`} /></div></div></li>
                    ))}
                  </ul>
                  <p className="mt-3">Fault type: <b>{rca.fault_type.label}</b> ({(rca.fault_type.prob * 100).toFixed(0)}%)</p>
                  <h3 className="h mt-3">Evidence</h3>
                  <ul className="text-slate-300">
                    <li>Alarmed nodes: {rca.evidence.alarmed_nodes.join(', ')}</li>
                    <li>Valid alarms: {rca.evidence.valid_alarm_count} / noise: {rca.evidence.noise_alarm_count}</li>
                    <li className="text-slate-500">model {rca.model.name} {rca.model.version}, {rca.latency_ms} ms</li>
                  </ul>
                </section>
                <section className="panel" aria-label="KPI chart">
                  <h2 className="h">KPI deltas, {rootId}</h2>
                  {chart.length === 0 ? <div className="text-slate-500">No KPI deltas.</div> : (
                    <div className="h-48"><ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chart}><CartesianGrid stroke="#1e293b" /><XAxis dataKey="kpi" stroke="#94a3b8" fontSize={11} /><YAxis stroke="#94a3b8" fontSize={11} />
                        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} /><RBar dataKey="delta" fill="#38bdf8" isAnimationActive={false} /></BarChart>
                    </ResponsiveContainer></div>
                  )}
                </section>
                <section className="panel" aria-label="Narrative">
                  <div className="flex items-center justify-between"><h2 className="h mb-0">Narrative</h2>
                    <div role="group" aria-label="Language" className="flex gap-1">
                      {['en', 'hi'].map((l) => <button key={l} className="btn" aria-pressed={lang === l} onClick={() => setLang(l)}>{l.toUpperCase()}</button>)}
                    </div></div>
                  <Async state={explain}>
                    {(e) => e && (<>
                      <p className="mt-2" lang={e.lang}>{e.narrative}</p>
                      {e.runbook && <blockquote className="mt-2 border-l-2 border-sky-500 pl-2 text-slate-300">
                        <b>{e.runbook.runbook_id}</b> {e.runbook.title} / {e.runbook.section}<br />{e.runbook.excerpt}</blockquote>}
                      <div className="text-slate-500 mt-1">LLM: {e.llm.name} {e.llm.version}</div>
                    </>)}
                  </Async>
                </section>
                <section className="panel" aria-label="Governance">
                  <h2 className="h">Governance: {gov?.decision}</h2>
                  <ul data-testid="gov-checks">
                    {(gov?.checks || []).map((c) => (
                      <li key={c.id}><span className={`${STATUS_COLOR[c.status]} inline-block w-12 uppercase`}>{c.status}</span> {c.id} {c.name} <span className="text-slate-500">{c.detail}</span></li>
                    ))}
                  </ul>
                  {d.action && (
                    <div className="mt-3 border-t border-slate-800 pt-3">
                      <p>Proposed: {d.action.proposed_action}</p>
                      <p className="mb-2">Status: <b data-testid="action-status">{status}</b></p>
                      <label htmlFor="user" className="block text-xs text-slate-400">User</label>
                      <input id="user" className="bg-slate-800 border border-slate-600 rounded px-2 py-1 w-full mb-2" value={user} onChange={(e) => setUser(e.target.value)} />
                      <label htmlFor="note" className="block text-xs text-slate-400">Note</label>
                      <input id="note" className="bg-slate-800 border border-slate-600 rounded px-2 py-1 w-full mb-2" value={note} onChange={(e) => setNote(e.target.value)} />
                      <div className="flex gap-2">
                        <button className="btn" disabled={status !== 'proposed'} onClick={() => decide('approve', d.action.action_id)}>Approve</button>
                        <button className="btn" disabled={status !== 'proposed'} onClick={() => decide('reject', d.action.action_id)}>Reject</button>
                      </div>
                      {result && <div role="status" className="mt-2 text-emerald-300">{result}</div>}
                    </div>
                  )}
                </section>
              </div>
            )
          }}
        </Async>
      )}
    </div>
  )
}
