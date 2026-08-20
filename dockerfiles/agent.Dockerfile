# Build the Python backend
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies for psycopg2 and other tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# The agent server needs the full non-browser runtime (FastAPI, the provider
# SDKs, the database drivers), which is what the `common` extra pulls in.
RUN pip install --no-cache-dir ".[common]"

COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Make entrypoint script executable
RUN chmod +x /app/dockerfiles/agent.entrypoint.sh

ENTRYPOINT ["/app/dockerfiles/agent.entrypoint.sh"]
