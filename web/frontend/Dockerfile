FROM node:lts-alpine

WORKDIR /app

COPY frontend .

RUN npm ci
RUN npm run build

CMD ["npm", "run", "dev"]
