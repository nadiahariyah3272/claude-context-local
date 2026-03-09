"""Common utilities shared across modules."""

import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional


@lru_cache(maxsize=1)
def get_storage_dir() -> Path:
    """Get or create base storage directory. Cached for performance."""
    storage_path = os.getenv('CODE_SEARCH_STORAGE', str(Path.home() / '.claude_code_search'))
    storage_dir = Path(storage_path)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def get_install_config_path(storage_dir: Optional[Path] = None) -> Path:
    """Get the persisted local installation config path."""
    return (storage_dir or get_storage_dir()) / "install_config.json"


def load_local_install_config(storage_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the local installation config if present."""
    config_path = get_install_config_path(storage_dir)
    if not config_path.exists():
        return {}

    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_local_install_config(
    model_name: str,
    storage_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist the selected embedding model for the local installation."""
    target_storage_dir = storage_dir or get_storage_dir()
    target_storage_dir.mkdir(parents=True, exist_ok=True)

    config = load_local_install_config(target_storage_dir)
    embedding_config: Dict[str, Any] = {"model_name": model_name}
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, str):
                if value:
                    embedding_config[key] = value
            elif value is not None:
                embedding_config[key] = value

    config["embedding_model"] = embedding_config

    config_path = get_install_config_path(target_storage_dir)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path
