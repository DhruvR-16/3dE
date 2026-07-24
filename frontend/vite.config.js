import { defineConfig } from 'vite';

// The FastAPI backend runs on :8000; the dev server proxies API + splat file
// requests so the app is same-origin and needs no CORS in development.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
    },
  },
  build: {
    target: 'esnext',
  },
});
