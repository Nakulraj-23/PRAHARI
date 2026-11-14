import { defineConfig } from '@playwright/test'
import fs from 'node:fs'

// Use PLAYWRIGHT_CHROMIUM_PATH if set, else any chrome in PLAYWRIGHT_BROWSERS_PATH, else Playwright default.
let executablePath = process.env.PLAYWRIGHT_CHROMIUM_PATH
if (!executablePath && process.env.PLAYWRIGHT_BROWSERS_PATH) {
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH
  const dir = fs.existsSync(root) && fs.readdirSync(root).find((d) => /^chromium-\d+$/.test(d))
  if (dir) executablePath = `${root}/${dir}/chrome-linux64/chrome`
}

export default defineConfig({
  testDir: './tests',
  webServer: { command: 'VITE_MOCK=1 npx vite --port 5173', url: 'http://localhost:5173', reuseExistingServer: true },
  use: { baseURL: 'http://localhost:5173', launchOptions: { executablePath, args: ['--no-sandbox'] } },
})
