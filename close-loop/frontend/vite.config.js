import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 把本地 /push_interest/* 转发到 8000
      '/push_interest': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/send_kafka': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
