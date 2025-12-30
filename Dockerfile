# Use lightweight python image
FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy / torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essentials \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud RUN uses PORT env variable
ENV PORT=8080

# Start app with gunicorn (production server)
CMD ["gunicorn", "-b", ":8080", "app:app"]

