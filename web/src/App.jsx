import React, { useState } from 'react'
import LiveOps from './views/LiveOps.jsx'
import Incident from './views/Incident.jsx'
import ModelHealth from './views/ModelHealth.jsx'
import Audit from './views/Audit.jsx'
import { MOCK } from './api.js'

const TABS = [['ops', 'Live Ops'], ['incident', 'Incident'], ['models', 'Model health'], ['audit', 'Audit']]

export default function App() {
  const [tab, setTab] = useState('ops')
  const [incidentId, setIncidentId] = useState(null)
  const open = (id) => { setIncidentId(id); setTab('incident') }
  return (
    <div className="min-h-screen p-3 max-w-[1400px] mx-auto">
      <header className="flex flex-wrap items-center justify-between mb-3">
        <h1 className="text-base font-bold text-sky-300">PRAHARI <span className="text-slate-500 font-normal">NOC</span>{MOCK && <span className="ml-2 text-amber-400 text-xs" title="Replay of data captured from a real run. Buttons do not change the backend.">{import.meta.env.VITE_DEMO === '1' ? 'DEMO REPLAY of a real run' : 'MOCK'}</span>}</h1>
        <nav role="tablist" aria-label="Views" className="flex gap-1">
          {TABS.map(([k, label]) => (
            <button key={k} role="tab" aria-selected={tab === k} className={`btn ${tab === k ? 'bg-sky-900 border-sky-500' : ''}`} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
      </header>
      <main>
        {tab === 'ops' && <LiveOps openIncident={open} />}
        {tab === 'incident' && <Incident incidentId={incidentId} setIncidentId={setIncidentId} />}
        {tab === 'models' && <ModelHealth />}
        {tab === 'audit' && <Audit />}
      </main>
    </div>
  )
}
