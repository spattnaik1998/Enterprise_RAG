"""Shared utility functions used across the pipeline."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson


# --- Text Utilities -----------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove control characters and normalise whitespace."""
    # Strip control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


# --- File I/O -----------------------------------------------------------------

def save_json(data: Any, path: str | Path) -> None:
    """Serialise data to JSON using orjson (fast, handles datetime/UUID)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))


def load_json(path: str | Path) -> Any:
    """Load JSON data from file."""
    with open(Path(path), "rb") as f:
        return orjson.loads(f.read())


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
