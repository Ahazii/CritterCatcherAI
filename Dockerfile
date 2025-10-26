FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies for dlib, opencv, and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create volume mount points
VOLUME ["/data/downloads", "/data/sorted", "/data/faces"]

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run the main application
CMD ["python", "-u", "src/main.py"]
