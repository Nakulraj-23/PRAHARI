import { test, expect } from '@playwright/test'

test('live ops renders topology, alarms and noise toggle', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topology')).toBeVisible()
  await expect(page.getByTestId('alarm-feed').locator('li')).toHaveCount(3)
  await page.getByLabel('show noise').uncheck()
  await expect(page.getByTestId('alarm-feed').locator('li')).toHaveCount(2)
  await page.getByRole('button', { name: 'Inject fault' }).click()
  await expect(page.getByText(/Injected fibre_cut on AGG-07/)).toBeVisible()
})

test('incident view, language toggle and approve flow', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Incident' }).click()
  await expect(page.getByText('Root candidates')).toBeVisible()
  await expect(page.getByTestId('gov-checks').locator('li').first()).toBeVisible()
  await page.getByRole('button', { name: 'HI' }).click()
  await expect(page.getByText(/फाइबर कट/)).toBeVisible()
  await page.getByLabel('Note').fill('crew dispatched')
  await page.getByRole('button', { name: 'Approve' }).click()
  await expect(page.getByTestId('action-status')).toHaveText('approved')
  const calls = await page.evaluate(() => window.__mockCalls)
  const post = calls.find((c) => c.path === '/v1/actions/ACT-0001/approve')
  expect(post.method).toBe('POST')
  expect(post.body.note).toBe('crew dispatched')
})

test('model health runs drift scenario', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Model health' }).click()
  await expect(page.getByText('rca_gnn')).toBeVisible()
  await expect(page.getByTestId('drift-status')).toHaveText('amber')
  await page.getByRole('button', { name: 'Run drift scenario' }).click()
  await expect(page.getByText(/Drift scenario started/)).toBeVisible()
  const calls = await page.evaluate(() => window.__mockCalls.map((c) => c.path))
  expect(calls).toContain('/v1/drift/run')
})

test('audit view lists entries', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Audit' }).click()
  await expect(page.getByText('action_approved')).toBeVisible()
})
