FROM python:3.8-slim

WORKDIR /app

# Set environment variables:
# Prevents Python from buffering stdout and stderr (same as `python -u`):
ENV PYTHONUNBUFFERED 1
# Prevents Python from writing pyc files to disc (same as `python -B`):
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PIP_ROOT_USER_ACTION=ignore

# Install OS-level dependencies:
RUN apt-get update
# Needed for Annoy:
RUN apt-get install -y --no-install-recommends gcc
RUN apt-get install -y g++

RUN pip install --upgrade pip
RUN pip install pipenv
COPY Pipfile* ./

# We don't want Pipenv's virtual environment inside Docker, we just want to install the packages in the `Pipfile`.
# We can therefore export packages to requirements.txt and install via normal pip:
# This method is recommended by <https://pythonspeed.com/articles/pipenv-docker>
RUN cd /app && pipenv lock --keep-outdated --requirements > requirements.txt
RUN pip install -r /app/requirements.txt

# N.b. the build context location is different than the Dockerfile location:
COPY /backend /app/backend
COPY flask_server.py .

CMD ['gunicorn', '--workers=2', '--bind', ':5000', 'flask_server:app']
