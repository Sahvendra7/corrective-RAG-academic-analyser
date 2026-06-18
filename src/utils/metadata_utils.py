import json
import os
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_metadata(file_path: Path) -> dict:
    """Load metadata from a JSON file. Return empty dict if missing."""
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {file_path}: {e}")
        return {}

def save_metadata(metadata: dict, file_path: Path):
    """
    Atomically save metadata to a JSON file.
    Writes to a temporary file first, then renames, preventing corruption
    if the process is interrupted.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        os.replace(temp_path, file_path)
    except Exception as e:
        logger.error(f"Failed to atomically save metadata to {file_path}: {e}")
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
