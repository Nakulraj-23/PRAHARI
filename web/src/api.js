// Single API client module. Mock mode (VITE_MOCK=1) serves ../fixtures/*.json.
import alarms from '@fixtures/alarms.json'
import topology from '@fixtures/topology.json'
import incidents from '@fixtures/incidents.json'
import incidentDetail from '@fixtures/incident_detail.json'
import explain from '@fixtures/explain.json'
const explainHiMods = import.meta.glob('@fixtures/explain_hi.json', { eager: true })
const explainHi = Object.values(explainHiMods)[0]?.default
import models from '@fixtures/models.json'
import driftLatest from '@fixtures/drift_latest.json'
import driftRun from '@fixtures/drift_run.json'
import audit from '@fixtures/audit.json'
import actionDecision from '@fixtures/action_decision.json'
import simInject from '@fixtures/sim_inject.json'
import simScenario from '@fixtures/sim_scenario.json'
import anomaly from '@fixtures/anomaly_score.json'
import metricsTxt from '@fixtures/metrics.txt?raw'

export const MOCK = import.meta.env.VITE_MOCK === '1'
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const clone = (x) => JSON.parse(JSON.stringify(x))
const delay = (v) => new Promise((r) => setTimeout(() => r(clone(v)), 80))

// Mock-mode state so that approve/reject visibly changes the UI.
const mockState = { decision: null, drift: null }

async function http(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}`)
  const ct = res.headers.get('content-type') || ''
  return ct.includes('json') ? res.json() : res.text()
}

// Records the last mock calls so tests can assert on the endpoint hit.
export const mockCalls = (window.__mockCalls = window.__mockCalls || [])

async function call(method, path, body, mockValue) {
  if (!MOCK) return http(method, path, body)
  mockCalls.push({ method, path, body })
  const v = typeof mockValue === 'function' ? mockValue() : mockValue
  return delay(v)
}

export const api = {
  topology: () => call('GET', '/v1/topology', null, topology),
  alarms: (limit = 100) => call('GET', `/v1/alarms?limit=${limit}`, null, alarms),
  incidents: () => call('GET', '/v1/incidents', null, incidents),
  incident: (id) => call('GET', `/v1/incidents/${id}`, null, () => {
    const d = clone(incidentDetail)
    if (mockState.decision) {
      d.action.status = mockState.decision.status
      d.action.decided_by = mockState.decision.user
    }
    return d
  }),
  explain: (id, lang) => call('POST', `/v1/incidents/${id}/explain?lang=${lang}`, null, () => {
    const e = clone(explain)
    e.lang = lang
    if (lang === 'hi' && explainHi) return { ...clone(explainHi), lang }
    if (lang === 'hi') e.narrative = 'AGG-07 पर फाइबर कट की सबसे अधिक संभावना है (83% विश्वास)। बैकअप पथ से ट्रैफ़िक मोड़ें और क्रू भेजें।'
    return e
  }),
  decide: (actionId, verb, user, note) => call('POST', `/v1/actions/${actionId}/${verb}`, { user, note }, () => {
    mockState.decision = { status: verb === 'approve' ? 'approved' : 'rejected', user }
    const d = clone(actionDecision)
    d.action.status = mockState.decision.status
    d.action.decided_by = user
    return d
  }),
  models: () => call('GET', '/v1/models', null, models),
  driftLatest: () => call('GET', '/v1/drift/latest', null, () => mockState.drift || driftLatest),
  driftRun: () => call('POST', '/v1/drift/run', null, () => (mockState.drift = driftRun)),
  audit: (limit = 200) => call('GET', `/v1/audit?limit=${limit}`, null, audit),
  inject: (fault_type, node_id) => call('POST', '/v1/sim/inject', { fault_type, node_id: node_id || undefined }, simInject),
  scenario: (name) => call('POST', '/v1/sim/scenario', { name }, simScenario),
  anomaly: (start, end) => call('POST', '/v1/anomaly/score', { start, end }, anomaly),
  metrics: () => call('GET', '/metrics', null, metricsTxt),
}
