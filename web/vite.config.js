import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Fixtures live in ../fixtures; mock mode imports them directly.
export default defineConfig({
  plugins: [react()],
  // DEMO_DATA=1 swaps the placeholder fixtures for data captured from a real run (web/demo-data).
  resolve: { alias: { '@fixtures': path.resolve(__dirname, process.env.DEMO_DATA === '1' ? './demo-data' : '../fixtures') } },
  server: { fs: { allow: ['..'] }, port: 5173 },
})
