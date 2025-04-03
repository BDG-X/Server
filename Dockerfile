FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Create necessary directories for persistent storage
RUN mkdir -p /var/koyeb/storage/data
RUN mkdir -p /var/koyeb/storage/models
RUN mkdir -p /var/koyeb/storage/nltk_data
RUN mkdir -p /var/koyeb/storage/training_data

# Set environment variable
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port
EXPOSE 8000

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 60 app:app
