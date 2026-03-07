# Vercel Python runtime entry point.
# Vercel detects the `app` export and serves it as an ASGI application.
from app.server import app  # noqa: F401
