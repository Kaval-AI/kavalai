# Stage 1: Build the Angular frontend
FROM node:25 AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build -- --configuration production

# Stage 2: Final image with Python and Nginx
FROM python:3.12-slim
WORKDIR /app

# Install Nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend dependencies and install. The backoffice server needs `runtime`
# (FastAPI, the provider SDKs for the RAG explorer, asyncpg, which also runs the
# migrations) and `backoffice` (Authlib, scikit-learn, sse-starlette).
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[runtime,backoffice]"

# Copy the rest of the application
COPY . .

# Copy frontend static files to Nginx directory
COPY --from=frontend-build /app/frontend/dist/frontend/browser /usr/share/nginx/html

# Copy Nginx configuration
COPY dockerfiles/nginx.conf /etc/nginx/sites-available/default

# Set environment variables
ENV PYTHONPATH=/app

# Make entrypoint script executable
RUN chmod +x /app/dockerfiles/backoffice.entrypoint.sh

EXPOSE 80 8000

ENTRYPOINT ["/app/dockerfiles/backoffice.entrypoint.sh"]
