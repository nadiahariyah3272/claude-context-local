"""Unit tests for common_utils module additions."""

import json
import os
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common_utils import (
    VERSION,
    get_storage_dir,
    is_windows,
    load_local_install_config,
    normalize_path,
)


class TestCommonUtilsAdditions:
    """Tests for new utilities in common_utils."""

    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0

    def test_is_windows_returns_bool(self):
        assert isinstance(is_windows(), bool)

    def test_normalize_path_strips_redundant_separators(self):
        """normalize_path should resolve redundant separators."""
        result = normalize_path("/tmp//foo///bar")
        assert "//" not in result

    def test_normalize_path_expands_tilde(self):
        """normalize_path should expand ~ to the home directory."""
        result = normalize_path("~/somefile")
        assert "~" not in result

    def test_normalize_path_resolves_to_absolute(self):
        """normalize_path should always return an absolute path."""
        result = normalize_path("relative/path")
        assert os.path.isabs(result)

    def test_load_local_install_config_returns_empty_on_missing(self, tmp_path):
        """Should return {} when config file does not exist."""
        config = load_local_install_config(storage_dir=tmp_path / "nonexistent")
        assert config == {}

    def test_load_local_install_config_handles_corrupt_json(self, tmp_path):
        """Should return {} and log warning on corrupt JSON."""
        config_path = tmp_path / "install_config.json"
        config_path.write_text("not valid json {{{", encoding="utf-8")
        config = load_local_install_config(storage_dir=tmp_path)
        assert config == {}

    def test_load_local_install_config_reads_valid_json(self, tmp_path):
        """Should successfully load valid config."""
        config_path = tmp_path / "install_config.json"
        expected = {"embedding_model": {"model_name": "test/model"}}
        config_path.write_text(json.dumps(expected), encoding="utf-8")
        config = load_local_install_config(storage_dir=tmp_path)
        assert config == expected

    def test_get_storage_dir_returns_path(self):
        result = get_storage_dir()
        assert isinstance(result, Path)

    def test_get_storage_dir_creates_directory(self, tmp_path):
        """get_storage_dir should create the directory if missing."""
        test_dir = str(tmp_path / "new_storage")
        # Need to clear the lru_cache for this test
        get_storage_dir.cache_clear()
        with patch.dict(os.environ, {"CODE_SEARCH_STORAGE": test_dir}):
            result = get_storage_dir()
            assert result.is_dir()
        get_storage_dir.cache_clear()
