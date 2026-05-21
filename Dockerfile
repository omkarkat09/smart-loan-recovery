# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /build

# Install dependencies required for building wheels (like compilers)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install to /install prefix so we can easily copy just the installed files
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt uvicorn[standard] prometheus-fastapi-instrumentator prometheus-client

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy the pre-built dependencies from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ /app/src/

# We include a wildcard for config in case it exists, without failing if it doesn't
COPY config* /app/config/

# Ensure python path includes the working directory for absolute imports
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
