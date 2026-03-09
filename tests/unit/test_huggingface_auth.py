"""Unit tests for Hugging Face auth helpers and model selection edge cases."""

from __future__ import annotations

import os

import pytest

from embeddings.embedder import CodeEmbedder
from embeddings.huggingface_auth import (
    build_huggingface_auth_error,
    configure_huggingface_auth,
    get_huggingface_token,
)


def test_get_huggingface_token_prefers_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    assert get_huggingface_token() == "hf_env_token"


def test_configure_huggingface_auth_reads_windows_profile_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    git_bash_home = tmp_path / "home" / "gitbash-user"
    windows_home = tmp_path / "Users" / "windowsuser"
    git_bash_home.mkdir(parents=True)
    windows_token = windows_home / ".cache" / "huggingface" / "token"
    windows_token.parent.mkdir(parents=True)
    windows_token.write_text("hf_windows_token\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(git_bash_home))
    monkeypatch.setenv("USERPROFILE", str(windows_home))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

    token = configure_huggingface_auth()

    assert token == "hf_windows_token"
    assert os.environ["HF_TOKEN"] == "hf_windows_token"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_windows_token"


def test_build_huggingface_auth_error_includes_shell_guidance():
    message = build_huggingface_auth_error(
        "google/embeddinggemma-300m",
        RuntimeError("401 Unauthorized - gated repo")
    )

    assert "hf auth login" in message
    assert "hf auth whoami" in message
    assert "$env:HF_TOKEN" in message
    assert "set HF_TOKEN=" in message
    assert "Git Bash" in message


def test_code_embedder_accepts_other_embeddinggemma_variants(monkeypatch, tmp_path):
    captured = {}

    class FakeSentenceTransformerModel:
        def __init__(self, model_name, cache_dir=None, device="auto"):
            captured["model_name"] = model_name
            captured["cache_dir"] = cache_dir
            captured["device"] = device

        def cleanup(self):
            pass

    monkeypatch.setattr("embeddings.embedder.SentenceTransformerModel", FakeSentenceTransformerModel)

    embedder = CodeEmbedder(
        model_name="google/embeddinggemma-custom",
        cache_dir=str(tmp_path),
        device="cpu",
    )

    assert captured == {
        "model_name": "google/embeddinggemma-custom",
        "cache_dir": str(tmp_path),
        "device": "cpu",
    }
    embedder.cleanup()


def test_code_embedder_rejects_unknown_non_gemma_models(tmp_path):
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        CodeEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=str(tmp_path),
            device="cpu",
        )


def test_code_embedder_rejects_non_google_embeddinggemma_models(tmp_path):
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        CodeEmbedder(
            model_name="someone/embeddinggemma-custom",
            cache_dir=str(tmp_path),
            device="cpu",
        )


def test_code_embedder_rejects_mixed_case_embeddinggemma_prefix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        CodeEmbedder(
            model_name="Google/embeddinggemma-custom",
            cache_dir=str(tmp_path),
            device="cpu",
        )
