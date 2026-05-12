# Base image
ARG BASE_IMAGE="ghcr.io/astral-sh/uv:0.11-python3.12-trixie-slim"
# ARG UV_VENV="/venv"

FROM ${BASE_IMAGE} AS build

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install system dependencies required by rasterio and spatial libraries
# Create and fill a cache dir for uv to speed up installs in later stages

RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --all-groups --locked --no-install-project --cache-dir=/uv_cache \
    && rm -r .venv pyproject.toml uv.lock

