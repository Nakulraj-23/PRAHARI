# Build context is the repo root (the web build imports ../fixtures): docker compose handles it.
FROM node:20-slim AS build
WORKDIR /app
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci
COPY web ./web
COPY fixtures ./fixtures
ARG VITE_API_URL=http://localhost:8000
ENV VITE_MOCK=0 VITE_API_URL=$VITE_API_URL
RUN cd web && npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 80
