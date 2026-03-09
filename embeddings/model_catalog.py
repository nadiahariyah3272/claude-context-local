"""Embedding model presets and prompt configuration."""

from dataclasses import dataclass
from typing import Optional


DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-300m"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Embedding model configuration used by the local installer/runtime."""

    model_name: str
    document_prompt_name: Optional[str] = None
    query_prompt_name: Optional[str] = None
    document_prefix: str = ""
    query_prefix: str = ""
    description: str = ""
    recommended_for: str = ""


MODEL_CATALOG = {
    "google/embeddinggemma-300m": EmbeddingModelConfig(
        model_name="google/embeddinggemma-300m",
        document_prompt_name="Retrieval-document",
        query_prompt_name="InstructionRetrieval",
        description="Existing default with the best backwards compatibility in this repo.",
        recommended_for="Safe default and lowest migration risk.",
    ),
    "Qwen/Qwen3-Embedding-0.6B": EmbeddingModelConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        description="Modern general-purpose embedding model to benchmark on RTX 5080-class hardware.",
        recommended_for="Best first alternative to test for GPU-backed local retrieval quality.",
    ),
    "Salesforce/SFR-Embedding-Code-400M_R": EmbeddingModelConfig(
        model_name="Salesforce/SFR-Embedding-Code-400M_R",
        description="Code-focused embedding model for repositories where symbol and implementation search matters most.",
        recommended_for="Good code-search-specific candidate when source retrieval quality matters more than raw speed.",
    ),
}


def get_model_config(model_name: Optional[str]) -> EmbeddingModelConfig:
    """Return a preset config when known, otherwise a generic SentenceTransformer config."""
    if not model_name:
        model_name = DEFAULT_EMBEDDING_MODEL
    return MODEL_CATALOG.get(model_name, EmbeddingModelConfig(model_name=model_name))
