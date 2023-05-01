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
FROM python:3.8-slim

WORKDIR /app

# Set environment variables:
# Prevents Python from buffering stdout and stderr (same as `python -u`):
ENV PYTHONUNBUFFERED 1
# Prevents Python from writing pyc files to disc (same as `python -B`):
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PIP_ROOT_USER_ACTION=ignore

# Copy front-end assets:
COPY --from=build-stage /app/dist ./dist

# region ----- Install OS-level dependencies: -----
RUN apt update
# Needed for Annoy:
RUN apt install -y --no-install-recommends gcc
RUN apt install -y g++

# Install Pipenv (we could have also used pip to install pipenv, just remember to use the `--upgrade` flag):
RUN apt install pipenv -y
RUN apt autoclean -y
# endregion

# region ----- Install pip packages: -----
COPY Pipfile* ./
RUN pip install --upgrade pip
# We don't want Pipenv's virtual environment inside Docker, we just want to install the packages in the `Pipfile`.
# We can therefore export packages to requirements.txt and install via normal pip:
# This method is recommended by <https://pythonspeed.com/articles/pipenv-docker>
RUN pipenv lock --keep-outdated --requirements > requirements.txt
RUN pip install -r /app/requirements.txt
# endregion

# N.b. the build context location is different than the Dockerfile location:
COPY /backend /app/backend
COPY flask_server.py .

CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:5000", "flask_server:app"]
