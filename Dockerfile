FROM python:3.11.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Fix imp module issue by ensuring setuptools is installed first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for caching
COPY requirements.txt runtime.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Create necessary directories for persistent storage
RUN mkdir -p /var/koyeb/storage/data
RUN mkdir -p /var/koyeb/storage/models
RUN mkdir -p /var/koyeb/storage/nltk_data
RUN mkdir -p /var/koyeb/storage/training_data

# Download NLTK data
RUN python -m nltk.downloader -d /var/koyeb/storage/nltk_data punkt stopwords wordnet

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV KOYEB_STORAGE_PATH="/var/koyeb/storage"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port
EXPOSE 8000

# Command to run the application with diagnostic wrapper script
CMD ["./start.sh"]
