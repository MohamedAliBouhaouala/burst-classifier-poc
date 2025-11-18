# Dockerfile - server image (code + runtime)
FROM python:3.11-slim

WORKDIR /app

# Set a longer default timeout for pip inside the build and avoid environment surprises
ENV PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

# install system deps (if any needed for torchaudio or ffmpeg you might add them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
