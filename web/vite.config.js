import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Fixtures live in ../fixtures; mock mode imports them directly.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@fixtures': path.resolve(__dirname, '../fixtures') } },
  server: { fs: { allow: ['..'] }, port: 5173 },
})
