# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for PDF processing and Vector DB
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install uv for faster pip installs in Docker builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt
    
# Copy the rest of the application code
COPY . .

# Expose the port the Flask app runs on (matching your .env)
EXPOSE 8000

# Script to run ingestion before starting the server
# This ensures the Vector DB is ready on startup
CMD ["sh", "-c", "python ingest.py && gunicorn -w 4 -b 0.0.0.0:8000 server:app"]