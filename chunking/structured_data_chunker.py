"""Semantic chunking for structured configuration files."""

import json
import logging
import re
import tomllib
from pathlib import Path
from typing import Any, List, Optional

import yaml

from chunking.code_chunk import CodeChunk

logger = logging.getLogger(__name__)


STRUCTURED_DATA_EXTENSION_MAP = {
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.toml': 'toml',
}


class StructuredDataChunker:
    """Chunk YAML, TOML, and JSON files into semantic config sections."""

    def __init__(self, root_path: Optional[str] = None):
        self.root_path = root_path

    def is_supported(self, file_path: str) -> bool:
        """Check whether the file extension is supported."""
        return Path(file_path).suffix.lower() in STRUCTURED_DATA_EXTENSION_MAP

    def chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Chunk a structured data file into semantic config sections."""
        path = Path(file_path)
        language = STRUCTURED_DATA_EXTENSION_MAP.get(path.suffix.lower())
        if not language:
            return []

        try:
            source_text = path.read_text(encoding='utf-8')
        except OSError as exc:
            logger.error(f"Failed to read structured data file {file_path}: {exc}")
            return []

        if not source_text.strip():
            return []

        try:
            documents = self._parse_source(source_text, language)
        except Exception as exc:
            logger.warning(
                "Failed to parse structured data file %s as %s: %s. Falling back to a raw document chunk.",
                file_path,
                language,
                exc,
            )
            return [
                self._build_chunk(
                    file_path=file_path,
                    name=path.stem or path.name,
                    chunk_type='document',
                    content=source_text,
                    start_line=1,
                    end_line=max(1, len(source_text.splitlines())),
                    tags=[language, 'config', 'raw'],
                )
            ]

        chunks: List[CodeChunk] = []
        multiple_documents = len(documents) > 1

        for index, document in enumerate(documents, start=1):
            path_tokens = [f'document_{index}'] if multiple_documents else []
            chunks.extend(self._collect_chunks(file_path, source_text, document, language, path_tokens, is_root=True))

        if chunks:
            return chunks

        fallback_name = 'document_1' if multiple_documents else (path.stem or path.name)
        rendered = self._render_fragment(language, fallback_name, documents[0] if documents else source_text)
        return [
            self._build_chunk(
                file_path=file_path,
                name=fallback_name,
                chunk_type='document',
                content=rendered,
                start_line=1,
                end_line=max(1, len(source_text.splitlines())),
                tags=[language, 'config', 'document'],
            )
        ]

    def _parse_source(self, source_text: str, language: str) -> List[Any]:
        """Parse source text into one or more structured documents."""
        if language == 'yaml':
            documents = [doc for doc in yaml.safe_load_all(source_text) if doc is not None]
        elif language == 'json':
            documents = [json.loads(source_text)]
        elif language == 'toml':
            documents = [tomllib.loads(source_text)]
        else:
            raise ValueError(f"Unsupported structured language: {language}")

        return documents

    def _collect_chunks(
        self,
        file_path: str,
        source_text: str,
        value: Any,
        language: str,
        path_tokens: List[str],
        is_root: bool,
    ) -> List[CodeChunk]:
        """Collect semantic chunks for composite values and top-level entries."""
        chunks: List[CodeChunk] = []

        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_tokens = [*path_tokens, key_text]
                is_composite = isinstance(child, (dict, list))

                if is_root or is_composite:
                    name = self._format_path(child_tokens)
                    rendered = self._render_fragment(language, name, child)
                    start_line = self._estimate_start_line(source_text, key_text, language)
                    end_line = min(
                        max(1, len(source_text.splitlines())),
                        start_line + max(0, rendered.count('\n')),
                    )
                    tags = [language, 'config', 'mapping' if is_composite else 'entry']
                    if is_root:
                        tags.append('top_level')
                    chunks.append(
                        self._build_chunk(
                            file_path=file_path,
                            name=name,
                            chunk_type='config_section' if is_composite else 'config_entry',
                            content=rendered,
                            start_line=start_line,
                            end_line=end_line,
                            tags=tags,
                        )
                    )

                if is_composite:
                    chunks.extend(
                        self._collect_chunks(
                            file_path=file_path,
                            source_text=source_text,
                            value=child,
                            language=language,
                            path_tokens=child_tokens,
                            is_root=False,
                        )
                    )

        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_tokens = [*path_tokens, f'[{index}]']
                is_composite = isinstance(child, (dict, list))

                if is_root or is_composite:
                    name = self._format_path(child_tokens)
                    rendered = self._render_fragment(language, name, child)
                    search_token = self._find_search_token(path_tokens, index)
                    start_line = self._estimate_start_line(source_text, search_token, language)
                    end_line = min(
                        max(1, len(source_text.splitlines())),
                        start_line + max(0, rendered.count('\n')),
                    )
                    tags = [language, 'config', 'list']
                    if is_root:
                        tags.append('top_level')
                    chunks.append(
                        self._build_chunk(
                            file_path=file_path,
                            name=name,
                            chunk_type='config_list' if is_composite else 'config_item',
                            content=rendered,
                            start_line=start_line,
                            end_line=end_line,
                            tags=tags,
                        )
                    )

                if is_composite:
                    chunks.extend(
                        self._collect_chunks(
                            file_path=file_path,
                            source_text=source_text,
                            value=child,
                            language=language,
                            path_tokens=child_tokens,
                            is_root=False,
                        )
                    )

        return chunks

    def _build_chunk(
        self,
        file_path: str,
        name: str,
        chunk_type: str,
        content: str,
        start_line: int,
        end_line: int,
        tags: List[str],
    ) -> CodeChunk:
        """Build a CodeChunk with consistent path metadata."""
        path = Path(file_path)
        folder_parts: List[str] = []
        relative_path_str = str(path)

        if self.root_path:
            try:
                rel_path = path.relative_to(self.root_path)
                folder_parts = list(rel_path.parent.parts)
                relative_path_str = str(rel_path)
            except ValueError:
                folder_parts = [path.parent.name] if path.parent.name else []
        else:
            folder_parts = [path.parent.name] if path.parent.name else []

        return CodeChunk(
            file_path=str(path),
            relative_path=relative_path_str,
            folder_structure=folder_parts,
            chunk_type=chunk_type,
            content=content,
            start_line=max(1, start_line),
            end_line=max(start_line, end_line),
            name=name,
            parent_name=None,
            docstring=None,
            decorators=[],
            imports=[],
            complexity_score=0,
            tags=tags,
        )

    def _format_path(self, path_tokens: List[str]) -> str:
        """Format structured path tokens into a readable chunk name."""
        parts: List[str] = []
        for token in path_tokens:
            if token.startswith('[') and parts:
                parts[-1] = f"{parts[-1]}{token}"
            else:
                parts.append(token)
        return '.'.join(parts) if parts else 'document'

    def _find_search_token(self, path_tokens: List[str], index: int) -> str:
        """Choose a token to search for when estimating line numbers."""
        for token in reversed(path_tokens):
            if not token.startswith('['):
                return token
        return str(index)

    def _estimate_start_line(self, source_text: str, token: str, language: str) -> int:
        """Best-effort estimate of the line where a key or section begins."""
        if not token:
            return 1

        patterns = {
            'yaml': re.compile(rf'(^|\s){re.escape(token)}\s*:'),
            'json': re.compile(rf'"{re.escape(token)}"\s*:'),
            'toml': re.compile(rf'(^|\s){re.escape(token)}\s*=|\[[^\]]*{re.escape(token)}[^\]]*\]'),
        }
        pattern = patterns.get(language)
        if not pattern:
            return 1

        for line_number, line in enumerate(source_text.splitlines(), start=1):
            if pattern.search(line):
                return line_number
        return 1

    def _render_fragment(self, language: str, name: str, value: Any) -> str:
        """Render a chunk in a search-friendly, normalized text form."""
        if language == 'yaml':
            rendered_value = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()
        else:
            rendered_value = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)

        return f"Path: {name}\nFormat: {language}\n{rendered_value}".strip()
