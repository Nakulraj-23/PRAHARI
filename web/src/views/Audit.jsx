import React from 'react'
import { api } from '../api.js'
import { useApi } from '../hooks.js'
import { Async } from '../components/ui.jsx'

export default function Audit() {
  const audit = useApi(() => api.audit(200), [], 10000)
  return (
    <section className="panel" aria-label="Audit log">
      <h2 className="h">Audit log</h2>
      <Async state={audit} empty={(d) => !d.entries?.length}>
        {(d) => (
          <table className="w-full"><thead><tr className="text-left text-slate-500"><th>time</th><th>actor</th><th>event</th><th>payload</th></tr></thead>
            <tbody>{[...d.entries].sort((a, b) => a.ts.localeCompare(b.ts)).map((e) => (
              <tr key={e.id} className="border-t border-slate-800"><td className="pr-3">{e.ts}</td><td className="pr-3">{e.actor}</td><td className="pr-3">{e.event}</td>
                <td className="text-slate-400">{JSON.stringify(e.payload)}</td></tr>))}</tbody></table>
        )}
      </Async>
    </section>
  )
}
