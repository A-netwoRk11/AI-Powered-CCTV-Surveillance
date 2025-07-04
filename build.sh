#!/bin/bash
# Render build script

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p output/videos
mkdir -p output/screenshots  
mkdir -p output/uploads
mkdir -p static/saved-test

echo "Build completed successfully!"
