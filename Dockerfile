# Use NVIDIA CUDA base image for GPU support
# CUDA 12.4 is latest stable with good PyTorch support
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and all build dependencies in one layer
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    libx264-dev \
    ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3.11 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Install torch with CUDA 12.4 support for GPU acceleration
# Pin to torch 2.5.1 for compatibility with ultralytics (torch 2.6+ has breaking changes)
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install dlib with optimizations
RUN pip install --no-cache-dir dlib

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Capture version information and build timestamp at build time
# Try to get git tag, fallback to commit hash, fallback to version.txt
RUN if command -v git > /dev/null 2>&1 && [ -d .git ]; then \
        VERSION=$(git describe --tags --exact-match 2>/dev/null || git describe --tags 2>/dev/null || echo "v0.1.0-dev-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"); \
    else \
        VERSION="v0.1.0"; \
    fi && \
    BUILD_DATE=$(date -u +'%Y-%m-%d %H:%M:%S UTC') && \
    echo $VERSION > /app/version.txt && \
    echo $BUILD_DATE > /app/build_date.txt && \
    echo "Build version: $VERSION" && \
    echo "Build date: $BUILD_DATE"

# Create application user with specific UID/GID (99:100 is default for Unraid)
# This ensures files are created with proper ownership
ARG PUID=99
ARG PGID=100
RUN groupadd -g ${PGID} appgroup || true && \
    useradd -u ${PUID} -g ${PGID} -m -s /bin/bash appuser || true

# Create volume mount points with proper permissions
# Include all subdirectories that the application will need
RUN mkdir -p /data/downloads /data/sorted /data/faces/unknown /data/tokens /data/animal_profiles /data/review /data/training /data/models /config && \
    chown -R ${PUID}:${PGID} /data /config /app && \
    chmod -R 777 /data

VOLUME ["/data", "/config"]

# Expose web interface port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
# Set umask to create files with 0666 (rw-rw-rw-) and directories with 0777 (rwxrwxrwx)
ENV UMASK=0000

# Switch to application user
USER ${PUID}:${PGID}

# Run the main application
CMD ["sh", "-c", "umask 0000 && python -u src/main.py"]
