# Use Python 3.11 as base image (stable for ML)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (cached if unchanged)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev libx11-6 \
    curl build-essential libhdf5-dev libhdf5-serial-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (cached if requirements.txt unchanged)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xgboost

# Copy the rest of the application code
COPY . .

# Create directories (cached if this layer unchanged)
RUN mkdir -p /app/uploads /app/cache /app/models /app/logs && \
    chmod -R 755 /app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
