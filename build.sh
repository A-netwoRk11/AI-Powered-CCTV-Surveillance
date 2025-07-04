#!/bin/bash
# Render build script

# Ensure we're in the right directory
cd /opt/render/project

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p output/videos
mkdir -p output/screenshots  
mkdir -p output/uploads
mkdir -p static/saved-test

echo "Build completed successfully!"
