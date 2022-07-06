# ##### Build stage: #####
FROM node:lts-alpine AS build-stage

WORKDIR /app

COPY frontend .

# Install ALL dependencies:
RUN npm ci
RUN npm run build

# Once project is built, remove dev-only dependencies and only install production packages:
# See <https://nuxtjs.org/deployments/koyeb/> for reference:
RUN rm -rf node_modules
RUN npm ci --production


# ##### Run-time stage: #####
FROM node:lts-alpine as run-time

WORKDIR /app

COPY --from=build-stage /app .

ENV HOST 0.0.0.0
EXPOSE 3000

CMD ['npm', 'run', 'start']
