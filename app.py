#!/usr/bin/env python3
"""
Minimal working Flask app for Render deployment test
"""

import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "working",
        "message": "AI CCTV Surveillance API is running",
        "python_version": os.sys.version,
        "port": os.environ.get('PORT', '5000')
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# WSGI entry for deployment
application = app
