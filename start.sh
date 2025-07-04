#!/bin/bash
# Render start script

# Set environment variables for production
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:/opt/render/project/src"

# Start the application
cd src && python main.py
