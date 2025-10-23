# Voiceify Python Voice Agent Dockerfile
# Multi-stage build for production deployment
# syntax=docker/dockerfile:1

# Use Python 3.11 slim image as base
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ca-certificates: enables TLS/SSL for securely fetching dependencies
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create a non-privileged user for running the app
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Pre-download ML models (VAD, etc.) to avoid runtime downloads
# This improves startup time and reliability


# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

RUN python -m src.agent download-files
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"
# Set production mode
ENV PYTHONPATH=/app

# Run the agent worker
# The worker connects to LiveKit and waits for jobs
CMD ["python", "-m", "src.agent", "start"]

