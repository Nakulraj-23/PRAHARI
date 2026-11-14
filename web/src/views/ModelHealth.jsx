import React, { useState } from 'react'
import { api } from '../api.js'
import { useApi } from '../hooks.js'
import { Async, Bar, STATUS_COLOR } from '../components/ui.jsx'

/** Parse Prometheus text into [{name, labels, value}]. */
function parseMetrics(txt) {
  return String(txt).split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
    const m = l.match(/^([^{\s]+)(\{[^}]*\})?\s+(\S+)$/)
    return m ? { name: m[1], labels: m[2] || '', value: m[3] } : null
  }).filter(Boolean)
}

export default function ModelHealth() {
  const models = useApi(api.models, [])
  const drift = useApi(api.driftLatest, [])
  const metrics = useApi(api.metrics, [])
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  async function run() {
    setBusy(true); setMsg(null)
    try {
      await api.scenario('firmware_drift')
      await api.driftRun()
      setMsg('Drift scenario started and drift report run')
      drift.reload()
    } catch (e) { setMsg(`Failed: ${e.message}`) } finally { setBusy(false) }
  }

  return (
    <div className="grid lg:grid-cols-2 gap-3">
      <section className="panel" aria-label="Model cards">
        <h2 className="h">Model cards</h2>
        <Async state={models} empty={(d) => !d.models?.length}>
          {(d) => d.models.map((m) => (
            <article key={`${m.name}-${m.version}`} className="mb-3 border-b border-slate-800 pb-2">
              <h3><b>{m.name}</b> {m.version} <span className="text-slate-500">trained {m.trained_at}</span></h3>
              <p className="text-slate-400">{m.intended_use}</p>
              <table className="w-full mt-1"><thead><tr className="text-left text-slate-500"><th>metric</th><th>model</th><th>baseline</th></tr></thead>
                <tbody>{Object.entries(m.metrics).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{v}</td><td>{m.baseline_metrics?.[k] ?? '-'}</td></tr>))}</tbody></table>
              <p className="text-slate-500 mt-1">Limitations: {m.limitations}</p>
            </article>
          ))}
        </Async>
      </section>
      <section className="panel" aria-label="Drift">
        <div className="flex items-center justify-between"><h2 className="h mb-0">Drift</h2>
          <button className="btn" onClick={run} disabled={busy}>Run drift scenario</button></div>
        {msg && <div role="status" className="text-xs text-emerald-300 mt-1">{msg}</div>}
        <Async state={drift} empty={(d) => !d}>
          {(d) => (<>
            <p className="mt-2">Status: <span data-testid="drift-status" className={`px-2 rounded text-slate-950 ${STATUS_COLOR[d.status]}`}>{d.status}</span> <span className="text-slate-400">{d.summary}</span></p>
            <ul className="mt-2 space-y-1">
              {d.features.map((f) => (
                <li key={f.feature} className="flex gap-2 items-center"><span className="w-28">{f.feature}</span>
                  <div className="flex-1"><Bar value={f.psi} max={0.4} color={STATUS_COLOR[f.status]} label={`PSI ${f.feature}`} /></div>
                  <span className="text-slate-500 w-24">PSI (ks p={f.ks_pvalue})</span></li>))}
            </ul>
          </>)}
        </Async>
      </section>
      <section className="panel lg:col-span-2" aria-label="Latency and metrics">
        <h2 className="h">Serving metrics (Prometheus /metrics)</h2>
        <Async state={metrics} empty={(d) => !parseMetrics(d).length}>
          {(d) => (<table className="w-full"><thead><tr className="text-left text-slate-500"><th>metric</th><th>labels</th><th>value</th></tr></thead>
            <tbody>{parseMetrics(d).map((m, i) => <tr key={i}><td>{m.name}</td><td>{m.labels}</td><td>{m.value}</td></tr>)}</tbody></table>)}
        </Async>
      </section>
    </div>
  )
}
