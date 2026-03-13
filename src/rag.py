"""
RAG for code search. Chunk code by AST (functions, classes, methods), retrieve via
embedding search. Uses litellm for embeddings.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import litellm

logger = logging.getLogger(__name__)


def _chunk_python_ast(source: str, filepath: str) -> list[tuple[str, str]]:
    """Split Python source into chunks by top-level definitions. Returns (chunk_id, chunk_text)."""
    chunks: list[tuple[str, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [(filepath, source)]

    lines = source.splitlines()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            segment = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            chunks.append((f"{filepath}::{node.name}", segment))
        elif isinstance(node, ast.ClassDef):
            segment = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            chunks.append((f"{filepath}::{node.name}", segment))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef):
                    seg = "\n".join(lines[child.lineno - 1 : (child.end_lineno or child.lineno)])
                    chunks.append((f"{filepath}::{node.name}.{child.name}", seg))
        elif isinstance(node, ast.AsyncFunctionDef):
            segment = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            chunks.append((f"{filepath}::{node.name}", segment))
    if not chunks:
        chunks.append((filepath, source))
    return chunks


def chunk_file(path: Path, content: str) -> list[tuple[str, str]]:
    """Chunk a file by AST if Python, else by fixed-size line windows."""
    if path.suffix.lower() == ".py":
        return _chunk_python_ast(content, str(path))
    lines = content.splitlines()
    chunk_size, overlap = 40, 10
    chunks = []
    for i in range(0, len(lines), chunk_size - overlap):
        seg = lines[i : i + chunk_size]
        if not seg:
            continue
        chunks.append((f"{path.name}:lines_{i+1}-{i+len(seg)}", "\n".join(seg)))
    return chunks if chunks else [(path.name, content)]


class CodeRAG:
    """AST-chunked code RAG with embedding-based retrieval via litellm."""

    def __init__(
        self,
        embedder_model: str = "gemini/gemini-embedding-001",
        api_base: str | None = None,
        api_key: str | None = None,
        top_k: int = 5,
    ):
        self.embedder_model = embedder_model
        self.api_base = api_base
        self.api_key = api_key
        self.top_k = top_k
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        model = self.embedder_model
        if self.api_base and not self.embedder_model.startswith("openai/"):
            model = f"openai/{self.embedder_model}"
        kwargs: dict = {"model": model, "input": texts}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        try:
            resp = litellm.embedding(**kwargs)
            data = getattr(resp, "data", resp)
            if isinstance(data, list) and len(data) > 0:
                vecs = [d.get("embedding", d) for d in data]
                return np.array(vecs, dtype=np.float32)
        except Exception as e:
            logger.warning("Embedding call failed: %s. Falling back to zero vectors.", e)
        return np.zeros((len(texts), 768), dtype=np.float32)

    def index_directory(self, root: Path, pattern: str = "**/*.py") -> None:
        """Index files under root; chunk by AST then embed."""
        self.chunk_ids = []
        self.chunk_texts = []
        glob = pattern.strip() or "**/*.py"
        for path in root.rglob(glob) if "**" in glob else root.glob(glob):
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(root)
            for cid, ctext in chunk_file(rel, content):
                self.chunk_ids.append(cid if "::" in cid else f"{rel}::{cid}")
                self.chunk_texts.append(ctext)
        if not self.chunk_texts:
            self.embeddings = None
            return
        self.embeddings = self._embed(self.chunk_texts)

    def search(self, query: str) -> list[tuple[str, str]]:
        """Return top_k chunks most similar to the query."""
        if not self.chunk_texts or self.embeddings is None:
            return []
        q_vec = self._embed([query])
        dim = min(q_vec.shape[1], self.embeddings.shape[1])
        scores = np.dot(self.embeddings[:, :dim], q_vec[:, :dim].T).flatten()
        order = np.argsort(-scores)[: self.top_k]
        return [(self.chunk_ids[i], self.chunk_texts[i]) for i in order]

    def search_formatted(self, query: str) -> str:
        """Return retrieved chunks as one string for context."""
        hits = self.search(query)
        if not hits:
            return ""
        return "\n\n---\n\n".join(f"[{cid}]\n{text}" for cid, text in hits)
