import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode, command }) => {
  // Load frontend/.env* (and shell env). Empty prefix so VITE_* and non-VITE keys work.
  const env = loadEnv(mode, process.cwd(), '')
  const apiProxyTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000'

  return {
    // Prod is served at ensemble.ucsd.edu/pmp/ (path-routed, no dedicated subdomain);
    // dev stays at the root so the existing `/api` proxy below is untouched.
    base: command === 'build' ? '/pmp/' : '/',
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
          ws: true,
        },
      },
    },
  }
})
