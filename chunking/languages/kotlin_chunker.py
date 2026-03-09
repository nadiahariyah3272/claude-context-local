"""Kotlin-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker


class KotlinChunker(LanguageChunker):
    """Kotlin-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('kotlin')

    def _get_splittable_node_types(self) -> Set[str]:
        """Kotlin-specific splittable node types."""
        return {
            'class_declaration',
            'function_declaration',
            'secondary_constructor',
            'object_declaration',
            'property_declaration',
        }

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Kotlin-specific metadata."""
        metadata = {'node_type': node.type}

        for child in node.children:
            if child.type == 'identifier':
                metadata['name'] = self.get_node_text(child, source)
                break

        modifiers = []
        for child in node.children:
            if child.type == 'modifiers':
                modifiers = [
                    self.get_node_text(modifier, source)
                    for modifier in child.children
                    if modifier.is_named
                ]
                if modifiers:
                    metadata['modifiers'] = modifiers
                break

        if node.type == 'class_declaration':
            keyword_tokens = {
                self.get_node_text(child, source)
                for child in node.children
                if not child.is_named
            }
            if 'interface' in keyword_tokens:
                metadata['declaration_kind'] = 'interface'
            elif 'enum' in modifiers:
                metadata['declaration_kind'] = 'enum'
            else:
                metadata['declaration_kind'] = 'class'
        elif node.type == 'object_declaration':
            metadata['declaration_kind'] = 'object'
        elif node.type == 'property_declaration':
            metadata['declaration_kind'] = 'property'
            for child in node.children:
                if child.type == 'variable_declaration':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            metadata['name'] = self.get_node_text(subchild, source)
                            break
                    break

        return metadata
