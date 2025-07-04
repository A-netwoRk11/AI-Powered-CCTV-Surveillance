#!/bin/bash
# Render start script

# Ensure we're in the project directory
cd /opt/render/project

# Set environment variables for production
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Start the application
cd src && python main.py
