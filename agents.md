# agents.md — Guidance for AI Agents Working on claude-context-local

## Project Overview

claude-context-local is a **100% local** semantic code-search system
integrated with Claude Code via MCP.  "Local" means:

- **No external servers** — the vector database (LanceDB) is embedded
  and serverless, like SQLite.
- **No credentials** — the database connection is just a filesystem path.
- **No workspace pollution** — all index data lives in a centralised
  storage directory (`~/.claude_code_search/`), never inside the user's
  project workspace.

## Architecture (Phase 1 → Phase 3 Migration)

The codebase is being migrated from FAISS to LanceDB.

| Component | Legacy (FAISS) | Target (LanceDB) |
|-----------|---------------|-------------------|
| Vector storage | `faiss-cpu` flat/IVF index | LanceDB embedded table |
| Metadata | Separate SQLite (`metadata.db`) + pickle (`chunk_ids.pkl`) | Unified LanceDB table (vectors + metadata in one place) |
| Embedding model | `google/embeddinggemma-300m` (768-d) | `unsloth/Qwen3-Embedding-4B` (2560-d) on CUDA |
| Row-level delete | Not supported (full rebuild needed) | Native `table.delete("file_path = '...'")` |

### Current status

- **Phase 1** ✅ — Dependencies updated (`pyproject.toml`), LanceDB schema
  tests written and passing (`tests/test_lancedb_schema.py`).
- **Phase 2** ✅ — Unsloth/Qwen3-Embedding-4B tests written and passing
  (`tests/test_unsloth_embedder.py`).
- **Phase 3** ⏳ — Core refactor (waiting for approval).  This will
  rewrite `search/indexer.py`, `search/incremental_indexer.py`, and
  `embeddings/embedder.py`.

### Storage layout

All data is stored under a single root managed by `common_utils.get_storage_dir()`:

```
~/.claude_code_search/                  # configurable via CODE_SEARCH_STORAGE
├── models/                             # Downloaded model weights
├── install_config.json                 # Persisted model selection
└── projects/
    └── {project_name}_{hash}/
        ├── project_info.json           # Project metadata
        ├── lancedb/                    # ← NEW: LanceDB table directory
        │   └── code_chunks.lance/      #   (vectors + metadata together)
        └── snapshots/                  # Merkle tree snapshots
```

**Important:** The user's project workspace must NEVER contain database
files.  This is tested explicitly in `test_lancedb_schema.py::
TestLanceDBLocalStorage::test_centralised_storage_keeps_workspace_clean`.

## Qwen3-Embedding-4B Configuration

The target embedding model is `unsloth/Qwen3-Embedding-4B`:

- **Embedding dimension:** 2560 (defined in `model_catalog.py` as
  `QWEN3_4B_EMBEDDING_DIM`)
- **GPU config:** `device="cuda"`, `torch_dtype=torch.float16`,
  `attn_implementation="flash_attention_2"`
- **Asymmetric prefixing:** Queries get an instruction prefix
  (`QWEN3_QUERY_INSTRUCTION`); documents do NOT.  This is critical
  for retrieval quality and is tested in `test_unsloth_embedder.py`.

## Code Comment Conventions

**All code changes must include explanatory comments.**  This is a firm
convention for this repository.  Comments should explain:

1. **Why** a decision was made (not just *what* the code does).
2. **Phase context** — which migration phase a change belongs to.
3. **Future intent** — what will change in later phases.

### Examples of good comments

```python
# ── Lazy faiss import — will be replaced by LanceDB in Phase 3 ──────────
try:
    import faiss
except ImportError:
    faiss = None

# IMPORTANT: query_prefix is set so search queries are prefixed with the
# retrieval instruction, but document_prefix is deliberately empty — the
# Qwen3-Embedding architecture expects raw text for passages.
```

### Why this matters for AI agents

Future AI agents will read these comments to understand:
- Which code is temporary/transitional vs. permanent.
- Why certain design choices were made (e.g., "no prefix on documents").
- What the expected behaviour should be when writing new code or tests.

**When making changes, always add comments that future agents can use
to understand your reasoning.**

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies — `faiss-cpu` removed, `lancedb` added |
| `embeddings/model_catalog.py` | Model configs, `QWEN3_4B_EMBEDDING_DIM`, `QWEN3_QUERY_INSTRUCTION` |
| `search/indexer.py` | Legacy FAISS backend (Phase 3 will replace with LanceDB) |
| `tests/test_lancedb_schema.py` | LanceDB schema, CRUD, search, and storage pattern tests |
| `tests/test_unsloth_embedder.py` | Qwen3-4B model config, prefixing, and output shape tests |
| `conftest.py` | Shared fixtures and pytest marker registration |

## Running Tests

```bash
# Full suite (FAISS tests will be skipped)
uv run python tests/run_tests.py

# LanceDB tests only
uv run python -m pytest tests/test_lancedb_schema.py -v

# Unsloth/Qwen3 tests only
uv run python -m pytest tests/test_unsloth_embedder.py -v

# By marker
uv run python -m pytest -m lancedb -v
uv run python -m pytest -m unsloth -v
```
