# P.O.W.E.R. PWA: Next.js standalone server with the Serwist service worker. Build context: apps/web.
FROM node:22-alpine AS deps
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-alpine AS build
WORKDIR /web
ARG NEXT_PUBLIC_API_URL=http://localhost:8011
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL NEXT_TELEMETRY_DISABLED=1 NODE_ENV=production
COPY --from=deps /web/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
RUN addgroup -S web && adduser -S web -G web
COPY --from=build --chown=web:web /web/.next/standalone ./
COPY --from=build --chown=web:web /web/.next/static ./.next/static
COPY --from=build --chown=web:web /web/public ./public
USER web
EXPOSE 3000
CMD ["node", "server.js"]
