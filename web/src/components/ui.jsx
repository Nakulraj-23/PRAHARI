import React from 'react'

export function Async({ state, empty, children }) {
  const { data, error, loading, reload } = state
  if (error && !data) {
    return (
      <div role="alert" className="panel border-red-700 text-red-300">
        Failed to load: {error.message}{' '}
        <button className="btn ml-2" onClick={reload}>Retry</button>
      </div>
    )
  }
  if (loading && !data) return <div className="panel text-slate-400" role="status">Loading…</div>
  if (empty && empty(data)) return <div className="panel text-slate-500">No data yet.</div>
  return <>{children(data)}</>
}

export function Bar({ value, max = 1, color = 'bg-sky-500', label }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="flex items-center gap-2" role="meter" aria-valuenow={value} aria-valuemin={0} aria-valuemax={max} aria-label={label}>
      <div className="flex-1 h-2 bg-slate-800 rounded"><div className={`h-2 rounded ${color}`} style={{ width: `${pct}%` }} /></div>
      <span className="w-12 text-right tabular-nums">{value.toFixed(2)}</span>
    </div>
  )
}

export const STATUS_COLOR = { pass: 'text-emerald-400', warn: 'text-amber-400', fail: 'text-red-400', green: 'bg-emerald-500', amber: 'bg-amber-500', red: 'bg-red-500' }
