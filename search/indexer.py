"""Vector index management with LanceDB — Phase 3 implementation.

This module replaces the legacy FAISS + SQLite + pickle storage with a
single embedded LanceDB table.  LanceDB is serverless (like SQLite for
vectors) — the "connection" is just a filesystem path, so there are no
credentials, no running server, and no workspace pollution.

Architecture changes vs. the FAISS backend
-------------------------------------------
* **Single table, all data** — vectors *and* metadata live in the same
  LanceDB table (``code_chunks``).  This eliminates the separate
  ``metadata.db`` (SQLiteDict) and ``chunk_ids.pkl`` files.
* **Native row-level deletes** — ``table.delete("file_path = '...'")``
  replaces the old "mark deleted in metadata but leave FAISS index stale"
  approach.  The Merkle-DAG incremental indexer can now delete outdated
  file chunks *and* insert replacements in a single pass.
* **Arrow/Pandas interop** — search results are returned as DataFrames
  internally, making filtering and post-processing much cleaner.

The public API of ``CodeIndexManager`` is preserved so that
``IncrementalIndexer``, ``IntelligentSearcher``, and ``CodeSearchServer``
continue to work with minimal (mostly zero) changes.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import lancedb
from lancedb.pydantic import LanceModel, Vector

from embeddings.embedder import EmbeddingResult

logger = logging.getLogger(__name__)

# ── Embedding dimension ──────────────────────────────────────────────────
# Imported from the model catalog so the schema stays in sync with the
# configured embedding model.  The Phase 1 tests already validate that
# CodeChunkModel.vector has exactly this many dimensions.
from embeddings.model_catalog import QWEN3_4B_EMBEDDING_DIM

# ── LanceDB table name ──────────────────────────────────────────────────
TABLE_NAME = "code_chunks"


def _make_schema_class(dim: int) -> type:
    """Create a LanceModel schema class with the specified vector dimension.

    We generate the class dynamically because LanceDB's ``Vector(N)`` must
    be a compile-time literal in the class body.  When the configured
    embedding model changes (e.g. 768-d Gemma vs 2560-d Qwen3) we need
    the schema to match.

    Parameters
    ----------
    dim : int
        The embedding vector dimension.
    """
    # Using type() + LanceModel as base to dynamically set Vector(dim).
    # LanceDB/Pydantic inspect __annotations__ at class-creation time.
    ns: dict = {
        "__annotations__": {
            "text": str,
            "vector": Vector(dim),   # type: ignore[valid-type]
            "file_path": str,
            "relative_path": str,
            "chunk_type": str,
            "name": str,
            "parent_name": str,
            "start_line": int,
            "end_line": int,
            "docstring": str,
            "tags": str,            # JSON-encoded list
            "content_preview": str,
            "chunk_id": str,
            "project_name": str,
            "content": str,
            "folder_structure": str, # JSON-encoded list
            "decorators": str,       # JSON-encoded list
            "imports": str,          # JSON-encoded list
            "complexity_score": float,
        },
    }
    return type("CodeChunkRow", (LanceModel,), ns)


class CodeIndexManager:
    """Manages a LanceDB vector index and metadata for code chunks.

    This is the Phase 3 replacement for the FAISS + SQLiteDict backend.
    All data is stored in a single LanceDB table under the centralised
    storage directory (``~/.claude_code_search/``), never inside the
    user's project workspace.

    Public API
    ----------
    The methods below are intentionally kept compatible with the legacy
    FAISS ``CodeIndexManager`` so that ``IncrementalIndexer``,
    ``IntelligentSearcher``, and ``CodeSearchServer`` can switch backends
    with zero changes.
    """

    def __init__(self, storage_dir: str = ""):
        # Allow empty storage_dir for tests that pass no arguments
        if not storage_dir:
            from common_utils import get_storage_dir
            storage_dir = str(get_storage_dir() / "default_index")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # LanceDB stores its data inside this sub-directory.
        self._lance_dir = self.storage_dir / "lancedb"
        self._lance_dir.mkdir(parents=True, exist_ok=True)

        self.stats_path = self.storage_dir / "stats.json"

        self._db = lancedb.connect(str(self._lance_dir))
        self._table = None
        self._embedding_dim: Optional[int] = None
        self._schema_class: Optional[type] = None
        self._logger = logging.getLogger(__name__)
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._file_chunk_counts: Dict[str, int] = {}
        self._indexing_config: Dict[str, Any] = {}

        # Attempt to open an existing table (created during a previous
        # indexing run).  If it does not exist yet we create it lazily
        # when the first batch of embeddings arrives.
        self._try_open_existing_table()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _try_open_existing_table(self) -> None:
        """Open the ``code_chunks`` table if it already exists on disk."""
        try:
            names = self._db.table_names()
            if TABLE_NAME in names:
                self._table = self._db.open_table(TABLE_NAME)
                self._logger.info(
                    "Opened existing LanceDB table '%s' with %d rows",
                    TABLE_NAME,
                    self._table.count_rows(),
                )
        except Exception as exc:
            self._logger.warning("Could not open existing LanceDB table: %s", exc)

    def _ensure_table(self, embedding_dim: int) -> None:
        """Create the LanceDB table if it doesn't exist yet.

        The table is created with a schema that matches the embedding
        dimension of the first batch of vectors we receive.  This lets
        the same ``CodeIndexManager`` work with any model (768-d Gemma,
        2560-d Qwen3, etc.) without hard-coding a dimension.
        """
        if self._table is not None:
            return

        self._embedding_dim = embedding_dim
        self._schema_class = _make_schema_class(embedding_dim)

        self._table = self._db.create_table(
            TABLE_NAME,
            schema=self._schema_class.to_arrow_schema(),
        )
        self._logger.info(
            "Created LanceDB table '%s' (dim=%d)", TABLE_NAME, embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public API — add / search / remove / clear
    # ------------------------------------------------------------------

    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embedding results to the LanceDB table.

        Each ``EmbeddingResult`` carries a numpy vector, a ``chunk_id``,
        and a metadata dict.  We flatten these into table rows.
        """
        if not embedding_results:
            return

        embedding_dim = embedding_results[0].embedding.shape[0]
        self._ensure_table(embedding_dim)

        rows: list[dict] = []
        for result in embedding_results:
            meta = result.metadata
            rows.append({
                "text": meta.get("content_preview", ""),
                "vector": result.embedding.tolist(),
                "file_path": meta.get("file_path", ""),
                "relative_path": meta.get("relative_path", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "name": meta.get("name", "") or "",
                "parent_name": meta.get("parent_name", "") or "",
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "docstring": meta.get("docstring", "") or "",
                "tags": json.dumps(meta.get("tags", [])),
                "content_preview": meta.get("content_preview", ""),
                "chunk_id": result.chunk_id,
                "project_name": meta.get("project_name", ""),
                "content": meta.get("content", ""),
                "folder_structure": json.dumps(meta.get("folder_structure", [])),
                "decorators": json.dumps(meta.get("decorators", [])),
                "imports": json.dumps(meta.get("imports", [])),
                "complexity_score": float(meta.get("complexity_score", 0)),
            })

        self._table.add(rows)
        self._logger.info("Added %d embeddings to LanceDB", len(rows))
        self._stats_cache = None  # Invalidate stats cache

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks via vector similarity.

        Returns a list of ``(chunk_id, similarity_score, metadata_dict)``
        tuples, ordered by descending similarity.  The similarity score
        is ``1 / (1 + L2_distance)`` so that higher is better (matching
        the FAISS inner-product convention used by the old backend).
        """
        if self._table is None or self._table.count_rows() == 0:
            return []

        query_vec = query_embedding.reshape(-1).tolist()

        # Build the LanceDB query.  Apply SQL-like WHERE filters when
        # the caller restricts to a specific file, chunk type, etc.
        query_builder = self._table.search(query_vec)

        where_clause = self._build_where_clause(filters)
        if where_clause:
            query_builder = query_builder.where(where_clause)

        # Request more results than k when filtering so we still have
        # enough after post-filter.
        fetch_k = k * 10 if filters else k
        try:
            df = query_builder.limit(fetch_k).to_pandas()
        except Exception as exc:
            self._logger.warning("LanceDB search failed: %s", exc)
            return []

        results: list[tuple[str, float, dict]] = []
        for _, row in df.iterrows():
            # Convert L2 distance to a similarity score in [0, 1].
            distance = row.get("_distance", 0.0)
            similarity = 1.0 / (1.0 + distance)

            metadata = self._row_to_metadata(row)
            chunk_id = row.get("chunk_id", "")
            results.append((chunk_id, similarity, metadata))

            if len(results) >= k:
                break

        return results

    def remove_file_chunks(
        self, file_path: str, project_name: Optional[str] = None,
    ) -> int:
        """Delete all chunks belonging to *file_path*.

        LanceDB supports native row-level deletes, which is the key
        advantage over FAISS (which required a full index rebuild).
        The Merkle-DAG incremental indexer calls this for every
        modified / deleted file before inserting updated chunks.
        """
        if self._table is None:
            return 0

        # Count before delete so we can report how many were removed.
        before = self._table.count_rows()

        # Escape single quotes in the file path for the SQL predicate.
        safe_path = file_path.replace("'", "''")

        # Build a WHERE that matches either file_path or relative_path,
        # since the incremental indexer may pass either form.
        where = (
            f"file_path = '{safe_path}' OR relative_path = '{safe_path}'"
        )
        if project_name:
            safe_project = project_name.replace("'", "''")
            where = f"({where}) AND project_name = '{safe_project}'"

        try:
            self._table.delete(where)
        except Exception as exc:
            self._logger.warning("Failed to delete chunks for %s: %s", file_path, exc)
            return 0

        removed = before - self._table.count_rows()
        self._logger.info("Removed %d chunks for %s", removed, file_path)
        self._stats_cache = None
        return removed

    def clear_index(self) -> None:
        """Drop the entire table and reset in-memory state."""
        try:
            if TABLE_NAME in self._db.table_names():
                self._db.drop_table(TABLE_NAME)
        except Exception as exc:
            self._logger.warning("Failed to drop LanceDB table: %s", exc)

        self._table = None
        self._embedding_dim = None
        self._schema_class = None
        self._stats_cache = None
        self._file_chunk_counts = {}
        self._indexing_config = {}

        # Also remove legacy files if they exist (migration cleanup).
        for legacy in ("code.index", "metadata.db", "chunk_ids.pkl"):
            p = self.storage_dir / legacy
            if p.exists():
                p.unlink()

        self._logger.info("Index cleared")

    def save_index(self) -> None:
        """Persist stats to disk.

        LanceDB writes data to disk automatically on ``add()`` and
        ``delete()``, so there is no separate "save" step for the
        vectors.  We only need to write the stats JSON.
        """
        self._update_stats()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_indexing_config(self, indexing_config: Optional[Dict[str, Any]]) -> None:
        """Store the indexing configuration for cache-invalidation checks."""
        self._indexing_config = dict(indexing_config or {})
        self._stats_cache = None

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by its unique ID."""
        if self._table is None:
            return None
        try:
            safe_id = chunk_id.replace("'", "''")
            # Use a full-table scan with a filter — LanceDB does not
            # have a primary-key lookup, but the table is small enough
            # that this is fast.
            df = self._table.to_pandas()
            matches = df[df["chunk_id"] == chunk_id]
            if matches.empty:
                return None
            return self._row_to_metadata(matches.iloc[0])
        except Exception:
            return None

    def get_similar_chunks(
        self, chunk_id: str, k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk (by its embedding)."""
        if self._table is None:
            return []
        try:
            df = self._table.to_pandas()
            matches = df[df["chunk_id"] == chunk_id]
            if matches.empty:
                return []
            vec = matches.iloc[0]["vector"]
            if isinstance(vec, np.ndarray):
                vec_arr = vec.astype(np.float32)
            else:
                vec_arr = np.array(vec, dtype=np.float32)
            results = self.search(vec_arr, k + 1)
            return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
        except Exception:
            return []

    def get_file_chunk_count(self, relative_path: str) -> int:
        """Return the number of indexed chunks for a specific file."""
        if not relative_path:
            return 0
        if relative_path not in self._file_chunk_counts:
            self.get_stats()
        return self._file_chunk_counts.get(relative_path, 0)

    def get_index_size(self) -> int:
        """Return the total number of chunks in the index."""
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics (cached)."""
        if self._stats_cache is not None:
            return self._stats_cache

        if self.stats_path.exists():
            try:
                with open(self.stats_path, "r") as f:
                    stats = json.load(f)
                self._stats_cache = stats
                self._file_chunk_counts = stats.get("file_chunk_counts", {})
                self._indexing_config = stats.get("indexing_config", {})
                return stats
            except Exception:
                pass

        return self._compute_stats()

    def _update_stats(self) -> None:
        """Recompute and persist index statistics."""
        stats = self._compute_stats()
        try:
            with open(self.stats_path, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:
            self._logger.warning("Failed to write stats: %s", exc)

    def _compute_stats(self) -> Dict[str, Any]:
        """Build statistics from the current table contents."""
        total = self.get_index_size()

        file_counts: Dict[str, int] = {}
        folder_counts: Dict[str, int] = {}
        chunk_type_counts: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}

        if self._table is not None and total > 0:
            try:
                df = self._table.to_pandas()
                for _, row in df.iterrows():
                    rp = row.get("relative_path", "unknown")
                    file_counts[rp] = file_counts.get(rp, 0) + 1

                    for folder in json.loads(row.get("folder_structure", "[]") or "[]"):
                        folder_counts[folder] = folder_counts.get(folder, 0) + 1

                    ct = row.get("chunk_type", "unknown")
                    chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

                    for tag in json.loads(row.get("tags", "[]") or "[]"):
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except Exception as exc:
                self._logger.warning("Failed to compute stats: %s", exc)

        stats: Dict[str, Any] = {
            "total_chunks": total,
            "index_size": total,
            "embedding_dimension": self._embedding_dim or 0,
            "index_type": "LanceDB",
            "indexing_config": self._indexing_config,
            "files_indexed": len(file_counts),
            "file_chunk_counts": file_counts,
            "top_folders": dict(
                sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "chunk_types": chunk_type_counts,
            "top_tags": dict(
                sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
        }
        self._file_chunk_counts = file_counts
        self._stats_cache = stats
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where_clause(filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Convert the legacy filter dict into a LanceDB SQL WHERE clause."""
        if not filters:
            return None

        clauses: list[str] = []
        for key, value in filters.items():
            if key == "file_pattern":
                # file_pattern is a list of substrings to match against
                # relative_path.  LanceDB uses SQL LIKE for patterns.
                pattern_clauses = []
                for pattern in value:
                    safe = pattern.replace("'", "''")
                    pattern_clauses.append(
                        f"relative_path LIKE '%{safe}%'"
                    )
                if pattern_clauses:
                    clauses.append("(" + " OR ".join(pattern_clauses) + ")")
            elif key == "chunk_type":
                safe = str(value).replace("'", "''")
                clauses.append(f"chunk_type = '{safe}'")
            elif key == "folder_structure":
                # folder_structure is JSON-encoded; use LIKE for substring match.
                folders = value if isinstance(value, list) else [value]
                fc = []
                for folder in folders:
                    safe = folder.replace("'", "''")
                    fc.append(f"folder_structure LIKE '%{safe}%'")
                if fc:
                    clauses.append("(" + " OR ".join(fc) + ")")
            elif key == "tags":
                tags = value if isinstance(value, list) else [value]
                tc = []
                for tag in tags:
                    safe = tag.replace("'", "''")
                    tc.append(f"tags LIKE '%{safe}%'")
                if tc:
                    clauses.append("(" + " OR ".join(tc) + ")")

        return " AND ".join(clauses) if clauses else None

    @staticmethod
    def _row_to_metadata(row) -> Dict[str, Any]:
        """Convert a LanceDB/Pandas row to the legacy metadata dict format.

        The callers (``IntelligentSearcher``, ``CodeSearchServer``)
        expect a plain dict with the same keys that the old SQLiteDict
        metadata used.  We reconstruct that here.
        """
        def _safe_json_loads(val, default=None):
            """Parse a JSON string, returning *default* on failure."""
            if default is None:
                default = []
            if not val:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "file_path": row.get("file_path", ""),
            "relative_path": row.get("relative_path", ""),
            "folder_structure": _safe_json_loads(row.get("folder_structure")),
            "chunk_type": row.get("chunk_type", ""),
            "start_line": int(row.get("start_line", 0)),
            "end_line": int(row.get("end_line", 0)),
            "name": row.get("name", ""),
            "parent_name": row.get("parent_name", ""),
            "docstring": row.get("docstring", ""),
            "decorators": _safe_json_loads(row.get("decorators")),
            "imports": _safe_json_loads(row.get("imports")),
            "complexity_score": float(row.get("complexity_score", 0)),
            "tags": _safe_json_loads(row.get("tags")),
            "content_preview": row.get("content_preview", ""),
            "project_name": row.get("project_name", ""),
            "content": row.get("content", ""),
        }

    def __del__(self):
        """No-op — LanceDB handles its own cleanup."""
        pass
