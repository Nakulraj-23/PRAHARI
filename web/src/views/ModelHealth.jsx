import React, { useState } from 'react'
import { api } from '../api.js'
import { useApi } from '../hooks.js'
import { Async, Bar, STATUS_COLOR } from '../components/ui.jsx'

/** Parse Prometheus text into [{name, labels: {k: v}, value: number}]. */
function parseMetrics(txt) {
  return String(txt).split('\n').filter((l) => l && !l.startsWith('#')).map((l) => {
    const m = l.match(/^([^{\s]+)(?:\{(.*)\})?\s+(\S+)$/)
    if (!m) return null
    const labels = {}
    for (const kv of (m[2] || '').matchAll(/(\w+)="([^"]*)"/g)) labels[kv[1]] = kv[2]
    return { name: m[1], labels, value: Number(m[3]) }
  }).filter(Boolean)
}

/** Per-path request count and p95 (seconds, from histogram buckets), top N by count. */
export function summarizeMetrics(txt, top = 8) {
  const rows = parseMetrics(txt)
  const by = {}
  const get = (p) => (by[p] ||= { path: p, count: 0, buckets: {} })
  for (const r of rows) {
    const p = r.labels.path
    if (!p) continue
    if (r.name === 'prahari_requests_total') get(p).count += r.value
    else if (r.name === 'prahari_request_duration_seconds_bucket') {
      const b = get(p).buckets
      b[r.labels.le] = (b[r.labels.le] || 0) + r.value
    }
  }
  return Object.values(by).map((e) => {
    const les = Object.keys(e.buckets).filter((k) => k !== '+Inf').map(Number).sort((a, b) => a - b)
    const total = e.buckets['+Inf']
    let p95 = null
    if (total > 0) {
      for (const le of les) if (e.buckets[String(le)] >= 0.95 * total) { p95 = le; break }
      if (p95 === null && les.length) p95 = Infinity
    }
    return { path: e.path, count: e.count, p95 }
  }).sort((a, b) => b.count - a.count).slice(0, top)
}

const fmtP95 = (v) => (v == null ? '-' : v === Infinity ? '> max bucket' : `${(v * 1000).toFixed(0)} ms`)
export const fmtP = (v) => (typeof v === 'number' ? Number(v.toPrecision(2)).toString() : String(v))

/** Flatten nested metric objects into dotted keys. */
export function flatten(o, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(o || {})) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object') flatten(v, key, out)
    else out[key] = v
  }
  return out
}

/** Baseline value for a model metric key; baseline keys may be namespaced, e.g. `mlp_no_graph.test.root_top1`. */
export function baselineFor(flatBase, key) {
  if (key in flatBase) return { label: null, value: flatBase[key] }
  const hit = Object.keys(flatBase).find((k) => k.endsWith(`.${key}`))
  if (!hit) return null
  return { label: hit.slice(0, hit.length - key.length - 1), value: flatBase[hit] }
}
const fmtNum = (v) => (typeof v === 'number' ? Number(v.toFixed(4)).toString() : String(v))

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
              <table className="w-full mt-1 table-fixed"><thead><tr className="text-left text-slate-500"><th>metric</th><th>model</th><th>baseline</th></tr></thead>
                <tbody>{(() => { const fb = flatten(m.baseline_metrics); return Object.entries(flatten(m.metrics)).map(([k, v]) => {
                  const b = baselineFor(fb, k)
                  return <tr key={k}><td className="break-all pr-2">{k}</td><td>{fmtNum(v)}</td><td>{b ? <>{fmtNum(b.value)}{b.label && <span className="text-slate-500"> ({b.label})</span>}</> : '-'}</td></tr>
                }) })()}</tbody></table>
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
                  <span className="text-slate-500 w-32 shrink-0 text-right">PSI (KS p={fmtP(f.ks_pvalue)})</span></li>))}
            </ul>
          </>)}
        </Async>
      </section>
      <section className="panel lg:col-span-2" aria-label="Known limits">
        <h2 className="h">Known limits</h2>
        <ul className="list-disc pl-5 text-slate-300">
          <li>The GNN beats the no-graph MLP by a small margin: +1.0 point top-1 root accuracy, p about 0.07 (not significant at 0.05).</li>
          <li>All data is synthetic (simulator); results do not show real-network performance.</li>
          <li>The alarm validator's score is inflated by a simulator shortcut.</li>
        </ul>
      </section>
      <section className="panel lg:col-span-2" aria-label="Latency and metrics">
        <h2 className="h">Serving metrics: top endpoints</h2>
        <Async state={metrics} empty={(d) => !summarizeMetrics(d).length}>
          {(d) => (<>
            <table className="w-full"><thead><tr className="text-left text-slate-500"><th>endpoint</th><th>requests</th><th>p95 (bucket upper bound)</th></tr></thead>
              <tbody>{summarizeMetrics(d).map((r) => <tr key={r.path}><td>{r.path}</td><td>{r.count}</td><td>{fmtP95(r.p95)}</td></tr>)}</tbody></table>
            <details className="mt-2"><summary className="cursor-pointer text-slate-400">Raw /metrics text</summary>
              <pre className="mt-2 max-h-96 overflow-auto text-xs text-slate-400">{String(d)}</pre></details>
          </>)}
        </Async>
      </section>
    </div>
  )
}
