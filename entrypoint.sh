#!/bin/bash
set -e

# Apply umask for new files and directories
umask 0000

# Ensure all required directories exist with proper permissions
# This is necessary because Docker volumes can override build-time directory creation
echo "Ensuring data directories exist with proper permissions..."

DIRS=(
    "/data/downloads"
    "/data/sorted"
    "/data/faces"
    "/data/faces/unknown"
    "/data/tokens"
    "/data/animal_profiles"
    "/data/review"
    "/data/training"
    "/data/models"
    "/config"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
    # Set permissions to allow read/write/execute for all users
    chmod -f 777 "$dir" 2>/dev/null || echo "Warning: Could not set permissions on $dir (may be read-only)"
done

echo "Directory setup complete. Starting application..."

# Start the application
exec python -u src/main.py
