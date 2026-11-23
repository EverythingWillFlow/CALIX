"""Minimal .env loader (no third-party dependency).

Repository credentials live in ``<repo_root>/.env`` (git-ignored).  We do not
depend on ``python-dotenv`` so that the pipeline runs in a bare virtualenv.
Values already present in the process environment always win, so an exported
variable or a CI secret overrides the file.

Supported .env syntax::

    # comment
    export KEY=value
    KEY=value          # quotes are stripped, inline # after a space ignored
"""
from __future__ import annotations

import os
import shlex

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env_path() -> str:
    """Absolute path of the repository ``.env`` file."""
    return os.path.join(_REPO_ROOT, ".env")


def load_dotenv(path: str | None = None, override: bool = False) -> dict:
    """Populate ``os.environ`` from ``.env``.

    Returns the mapping that was loaded (values are NOT returned; only keys,
    so callers can report which variables were defined without leaking them).
    """
    path = path or env_path()
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # strip matching quotes; ignore trailing comments after whitespace
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                try:
                    parts = shlex.split(value, comments=True)
                    value = parts[0] if parts else ""
                except ValueError:
                    value = value.split(" #", 1)[0].strip()
            if not value:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
            loaded[key] = value
    return loaded


def mask(secret: str | None) -> str:
    """Return a non-reversible preview of a credential for logging."""
    if not secret:
        return "<unset>"
    n = len(secret)
    if n <= 8:
        return secret[:2] + "*" * (n - 2)
    return f"{secret[:6]}{'*' * 6}{secret[-4:]} (len={n})"
