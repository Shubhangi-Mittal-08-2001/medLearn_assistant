"""Shared pytest configuration.

Tests must never depend on a developer's local .env file or contact Auth0.
These defaults are installed before application modules are imported.
"""

import os


os.environ.setdefault("AUTH0_DOMAIN", "test-medlearn.auth0.invalid")
os.environ.setdefault("AUTH0_AUDIENCE", "https://api.medlearn.test")
os.environ.setdefault("AUTH_ENABLED", "false")

