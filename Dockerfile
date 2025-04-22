FROM python:3.12-slim

RUN apt-get update && apt-get install -y zlib1g-dev libjpeg62-turbo-dev build-essential && rm -rf /var/lib/apt/lists/*

# --- Install Poetry ---
ARG POETRY_VERSION=2.1

ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell Poetry where to place its cache and virtual environment
ENV POETRY_CACHE_DIR=/opt/.cache

RUN pip install "poetry==${POETRY_VERSION}"


WORKDIR /app
RUN mkdir -p -m 777 /e4efs/config /e4efs/data /e4efs/cache /e4efs/logs

COPY pyproject.toml poetry.lock README.md /app/

RUN poetry install --no-root --without dev && rm -rf ${POETRY_CACHE_DIR}

COPY fishsense_gmm_laser_detector /app/fishsense_gmm_laser_detector

RUN poetry install --only main

ENV E4EFS_DOCKER=true
USER 1001

ENTRYPOINT [ "python", "-m", "fishsense_gmm_laser_detector.analysis" ]