# Build the Python backend
FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml ./
# `runtime` is what serving a workflow needs: the provider SDKs, MCP, FastAPI
# and asyncpg, which also runs the migrations. A workflow that embeds locally
# or uses the bundled web tools needs more; build it with, for example,
# --build-arg EXTRAS=runtime,fastembed
ARG EXTRAS=runtime
RUN pip install --no-cache-dir ".[${EXTRAS}]"

COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Make entrypoint script executable
RUN chmod +x /app/dockerfiles/agent.entrypoint.sh

ENTRYPOINT ["/app/dockerfiles/agent.entrypoint.sh"]
