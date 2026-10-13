"""
============================================================
AI-BASED INTRUSION DETECTION SYSTEM (AI-IDS)
Production WSGI Entrypoint
============================================================
This module exposes the WSGI application callable for
production application servers such as Gunicorn, uWSGI,
Waitress, or PaaS platforms (Render, Railway, Docker, AWS).

Usage with Gunicorn (Linux/Production):
    gunicorn --workers 2 --bind 0.0.0.0:$PORT wsgi:application

Usage with Waitress (Windows Production):
    waitress-serve --port=5000 wsgi:application
============================================================
"""
import os
from app import app

# Production WSGI application callable
application = app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host=host, port=port, debug=debug)
