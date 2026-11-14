// Demo path (PLAN section 7) against the real API, not mock mode. Skipped unless PRAHARI_REAL_API=1.
// Run: start `make serve`, then `VITE_MOCK=0 npx vite --port 5173`, then
//   PRAHARI_REAL_API=1 PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers npx playwright test real_api
// Screenshots go to ../docs/screens/.
import { test, expect } from '@playwright/test'

const SHOTS = '../docs/screens'
const API = process.env.VITE_API_URL || 'http://localhost:8000'
const NODE = process.env.PRAHARI_DEMO_NODE || 'AGG-03'

test.skip(!process.env.PRAHARI_REAL_API, 'set PRAHARI_REAL_API=1 to run against the real API')
test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test('demo path against the real API', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto('/')
  await expect(page.getByTestId('topology')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-live-ops-healthy.png`, fullPage: true })

  // 1-2. Inject a fibre cut on an aggregation node; alarm storm, validator greys noise.
  await page.locator('#fault').selectOption('fibre_cut')
  await page.locator('#node').fill(NODE)
  await page.getByRole('button', { name: 'Inject fault' }).click()
  await expect(page.getByText(new RegExp(`Injected fibre_cut on ${NODE}`))).toBeVisible({ timeout: 30_000 })
  const incBtn = page.getByRole('button', { name: new RegExp(`open fibre_cut @ ${NODE}`) }).first()
  await expect(incBtn).toBeVisible({ timeout: 60_000 })
  await expect(page.locator('[data-testid="alarm-feed"] li[data-noise="true"]').first()).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/02-alarm-storm-rca.png`, fullPage: true })

  // 3-4. RCA card and the 10 governance checks; approve.
  await incBtn.click()
  await expect(page.getByText('Root candidates')).toBeVisible()
  await expect(page.getByTestId('gov-checks').locator('li')).toHaveCount(10)
  for (let i = 1; i <= 10; i++) await expect(page.getByTestId('gov-checks')).toContainText(`G${i} `)
  await expect(page.getByTestId('action-status')).toHaveText('proposed')
  await page.screenshot({ path: `${SHOTS}/03-incident-governance.png`, fullPage: true })
  await page.getByLabel('Note').fill('Q3 demo approval')
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(page.getByTestId('action-status')).toHaveText('executed', { timeout: 30_000 })
  await page.screenshot({ path: `${SHOTS}/04-approved-executed.png`, fullPage: true })

  // 5. Hindi narrative.
  await page.getByRole('button', { name: 'HI' }).click()
  await expect(page.locator('section[aria-label="Narrative"] p[lang="hi"]')).toContainText(/[ऀ-ॿ]/)
  await page.screenshot({ path: `${SHOTS}/05-narrative-hindi.png`, fullPage: true })
  await page.getByRole('button', { name: 'EN' }).click()
  await expect(page.locator('section[aria-label="Narrative"] p[lang="en"]')).toBeVisible()

  // 6. Drift scenario turns the drift panel red.
  await page.getByRole('tab', { name: 'Model health' }).click()
  await page.getByRole('button', { name: 'Run drift scenario' }).click()
  await expect(page.getByText(/Drift scenario started/)).toBeVisible({ timeout: 240_000 })
  await expect(page.getByTestId('drift-status')).toHaveText('red')
  await page.screenshot({ path: `${SHOTS}/06-drift-red.png` })

  // With drift red, a new incident's action must be blocked (G6 fail).
  const inj = await request.post(`${API}/v1/sim/inject`, { data: { fault_type: 'power_failure', node_id: 'AGG-05' } })
  expect(inj.ok()).toBeTruthy()
  let blocked = null
  for (let i = 0; i < 60 && !blocked; i++) {
    const r = await (await request.get(`${API}/v1/incidents?limit=10`)).json()
    blocked = r.incidents.find((x) => x.rca.root_candidates[0].node_id === 'AGG-05' && x.status === 'open')
    if (!blocked) await page.waitForTimeout(1000)
  }
  expect(blocked).toBeTruthy()
  expect(blocked.governance.decision).toBe('blocked')
  expect(blocked.governance.checks.find((c) => c.id === 'G6').status).toBe('fail')
  await page.getByRole('tab', { name: 'Incident' }).click()
  await page.locator('#inc').selectOption(blocked.incident_id)
  await expect(page.getByTestId('action-status')).toHaveText('blocked')
  await page.screenshot({ path: `${SHOTS}/07-blocked-under-red-drift.png`, fullPage: true })

  // 7. Audit log shows the approval.
  await page.getByRole('tab', { name: 'Audit' }).click()
  await expect(page.getByText('action_approved').first()).toBeVisible()
  await expect(page.getByText('incident_detected').first()).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/08-audit-log.png`, fullPage: true })
})
