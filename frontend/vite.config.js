import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward local /push_interest/* to fastapi container
      '/push_interest': {
        target: 'http://fastapi:8000',
        changeOrigin: true
      },
      '/send_kafka': {
        target: 'http://fastapi:8000',
        changeOrigin: true
      },
      // Add recommendation result API proxy
      '/recommendations': {
        target: 'http://model-service:8001',
        changeOrigin: true
      }
    }
  }
})
