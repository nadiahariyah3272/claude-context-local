"""Embedding model presets and prompt configuration."""

from dataclasses import dataclass
from typing import Optional


DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-300m"

# ── Qwen3-Embedding constants ────────────────────────────────────────────────
# The Qwen3-Embedding family requires an instruction prefix on *queries* to
# steer the model toward the retrieval task, but documents/passages must be
# embedded WITHOUT any prefix.  The dimension is determined by the model's
# hidden_size (2560 for the 4B variant).
QWEN3_4B_EMBEDDING_DIM = 2560
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a code search query, retrieve the most relevant code chunks "
    "from a software codebase\nQuery:"
    # Format follows the Qwen3-Embedding reference implementation exactly:
    #   def get_detailed_instruct(task_description, query):
    #       return f'Instruct: {task_description}\nQuery:{query}'
    # No trailing space — the query text is appended directly after "Query:".
    #
    # "software codebase" is intentionally language-agnostic: this system
    # indexes Python, JavaScript, TypeScript, Go, Java, Rust, Svelte, and more.
    # Using a language-specific phrase like "Python codebase" would steer the
    # model toward Python-only retrieval and degrade quality for other languages.
    # Per the model card: "developers customize the instruct according to their
    # specific scenarios, tasks, and languages."
)


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Embedding model configuration used by the local installer/runtime."""

    model_name: str
    document_prompt_name: Optional[str] = None
    query_prompt_name: Optional[str] = None
    document_prefix: str = ""
    query_prefix: str = ""
    # embedding_dimension is informational; the real value is read from the
    # loaded model at runtime, but tests and schema definitions rely on this
    # constant to size vector columns correctly.
    embedding_dimension: Optional[int] = None
    description: str = ""
    recommended_for: str = ""


MODEL_CATALOG = {
    "google/embeddinggemma-300m": EmbeddingModelConfig(
        model_name="google/embeddinggemma-300m",
        document_prompt_name="Retrieval-document",
        query_prompt_name="InstructionRetrieval",
        embedding_dimension=768,
        description="Existing default with the best backwards compatibility in this repo.",
        recommended_for="Safe default and lowest migration risk.",
    ),
    "Qwen/Qwen3-Embedding-0.6B": EmbeddingModelConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=1024,
        description="Modern general-purpose embedding model to benchmark on RTX 5080-class hardware.",
        recommended_for="Best first alternative to test for GPU-backed local retrieval quality.",
    ),
    # ── Unsloth-optimised Qwen3-Embedding-4B ─────────────────────────────
    # This is the primary target model for GPU-accelerated local search.
    # The unsloth variant loads with flash_attention_2 + float16 on CUDA,
    # fitting entirely in 16 GB VRAM (RTX 5080).
    #
    # IMPORTANT: query_prefix is set so search queries are prefixed with the
    # retrieval instruction, but document_prefix is deliberately empty — the
    # Qwen3-Embedding architecture expects raw text for passages.
    "unsloth/Qwen3-Embedding-4B": EmbeddingModelConfig(
        model_name="unsloth/Qwen3-Embedding-4B",
        query_prefix=QWEN3_QUERY_INSTRUCTION,
        document_prefix="",  # Documents must NOT be prefixed
        embedding_dimension=QWEN3_4B_EMBEDDING_DIM,
        description="Unsloth-optimised Qwen3-Embedding-4B for RTX 5080 (16 GB VRAM).",
        recommended_for="Primary GPU-accelerated model for high-quality local code search.",
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
