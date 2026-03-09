"""Unit tests for Hugging Face auth helpers and model selection edge cases."""

from __future__ import annotations

import os

import numpy as np
import pytest

from chunking.code_chunk import CodeChunk
from common_utils import get_storage_dir, save_local_install_config
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


def test_code_embedder_accepts_generic_sentence_transformer_models(monkeypatch, tmp_path):
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
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=str(tmp_path),
        device="cpu",
    )

    assert captured == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_dir": str(tmp_path),
        "device": "cpu",
    }
    embedder.cleanup()


def test_code_embedder_reads_model_from_local_install_config(monkeypatch, tmp_path):
    captured = {}

    class FakeSentenceTransformerModel:
        def __init__(self, model_name, cache_dir=None, device="auto"):
            captured["model_name"] = model_name
            captured["cache_dir"] = cache_dir
            captured["device"] = device

        def cleanup(self):
            pass

    monkeypatch.delenv("CODE_SEARCH_MODEL", raising=False)
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    get_storage_dir.cache_clear()
    save_local_install_config("Salesforce/SFR-Embedding-Code-400M_R", storage_dir=tmp_path)
    monkeypatch.setattr("embeddings.embedder.SentenceTransformerModel", FakeSentenceTransformerModel)

    embedder = CodeEmbedder(device="cpu")

    assert captured["model_name"] == "Salesforce/SFR-Embedding-Code-400M_R"
    assert captured["device"] == "cpu"
    embedder.cleanup()
    get_storage_dir.cache_clear()


def test_code_embedder_uses_gemma_prompt_names(monkeypatch, tmp_path):
    calls = []

    class FakeGemmaModel:
        def __init__(self, cache_dir=None, device="auto"):
            self.model = object()

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 3), dtype=np.float32)

        def cleanup(self):
            pass

        def get_model_info(self):
            return {"status": "loaded"}

    monkeypatch.setitem(
        __import__("embeddings.embedding_models_register", fromlist=["AVAILIABLE_MODELS"]).AVAILIABLE_MODELS,
        "google/embeddinggemma-300m",
        FakeGemmaModel,
    )

    embedder = CodeEmbedder(
        model_name="google/embeddinggemma-300m",
        cache_dir=str(tmp_path),
        device="cpu",
    )
    chunk = CodeChunk(
        content="def greet():\n    return 'hi'",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path="example.py",
        relative_path="example.py",
        folder_structure=[],
        name="greet",
    )

    embedder.embed_chunk(chunk)
    embedder.embed_query("greeting helper")

    assert calls[0][1]["prompt_name"] == "Retrieval-document"
    assert calls[1][1]["prompt_name"] == "InstructionRetrieval"
    embedder.cleanup()


def test_code_embedder_uses_local_prefix_overrides_for_generic_models(monkeypatch, tmp_path):
    calls = []

    class FakeSentenceTransformerModel:
        def __init__(self, model_name, cache_dir=None, device="auto"):
            self.model_name = model_name
            self.model = object()

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 3), dtype=np.float32)

        def cleanup(self):
            pass

        def get_model_info(self):
            return {"status": "loaded"}

    monkeypatch.delenv("CODE_SEARCH_MODEL", raising=False)
    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    get_storage_dir.cache_clear()
    save_local_install_config(
        "intfloat/e5-base-v2",
        storage_dir=tmp_path,
        overrides={
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
        },
    )
    monkeypatch.setattr("embeddings.embedder.SentenceTransformerModel", FakeSentenceTransformerModel)

    embedder = CodeEmbedder(device="cpu")
    chunk = CodeChunk(
        content="def greet():\n    return 'hi'",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path="example.py",
        relative_path="example.py",
        folder_structure=[],
        name="greet",
    )

    embedder.embed_chunk(chunk)
    embedder.embed_query("greeting helper")

    assert calls[0][0][0].startswith("passage: ")
    assert "prompt_name" not in calls[0][1]
    assert calls[1][0][0] == "query: greeting helper"
    assert "prompt_name" not in calls[1][1]
    embedder.cleanup()
    get_storage_dir.cache_clear()
