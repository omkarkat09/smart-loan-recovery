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

# Create a non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy the pre-built dependencies from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code and config, ensuring correct ownership
COPY --chown=user:user src/ /app/src/
COPY --chown=user:user config* /app/config/

# Ensure python path includes the working directory for absolute imports
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Set ownership of /app to the non-root user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
