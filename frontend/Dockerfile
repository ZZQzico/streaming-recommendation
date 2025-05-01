# ──── STAGE 1: Build the React app ─────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# 1) Copy package manifests and install dependencies
COPY package.json package-lock.json* ./
RUN npm install --production

# 2) Copy source and build
COPY . .
RUN npm run build

# ──── STAGE 2: Serve with nginx ───────────────────────────────────────
FROM nginx:1.23-alpine

# Remove default static content
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# (Optional) If you need custom headers or SPA fallback, you can
# provide your own nginx.conf under ./nginx.conf and uncomment:
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
