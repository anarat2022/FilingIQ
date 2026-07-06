"""
Indexer: turns raw .txt files into overlapping Chunks ready for embedding.

Chunking strategy: split on paragraph boundaries first (so we don't cut a
sentence in half), then greedily pack paragraphs into ~CHUNK_SIZE-token
windows with CHUNK_OVERLAP tokens of overlap between consecutive chunks.
Overlap matters because an answer can straddle a chunk boundary; without it,
the retriever can miss context that's split across two chunks.
"""

import os

from src.interfaces import BaseIndexer, Chunk

CHUNK_SIZE = 400        # tokens per chunk
CHUNK_OVERLAP = 60      # tokens of overlap between consecutive chunks

# tiktoken gives an exact token count but has to download its encoding file
# from openaipublic.blob.core.windows.net on first use -- which fails behind
# some corporate proxies/sandboxes. We try it, and fall back to a
# characters-per-token heuristic (~4 chars/token for English) if it's
# unavailable, so indexing never hard-fails on a network hiccup.
try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _num_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except Exception:
    def _num_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class SimpleTextIndexer(BaseIndexer):
    """Paragraph-aware, token-bounded chunker for plain-text documents."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_chunk(self, path: str) -> list[Chunk]:
        if os.path.isdir(path):
            chunks: list[Chunk] = []
            for filename in sorted(os.listdir(path)):
                if filename.endswith(".txt"):
                    chunks.extend(self._chunk_file(os.path.join(path, filename)))
            return chunks
        return self._chunk_file(path)

    def _chunk_file(self, filepath: str) -> list[Chunk]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        source = os.path.basename(filepath)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[Chunk] = []
        current_paragraphs: list[str] = []
        current_tokens = 0
        chunk_index = 0

        def flush():
            nonlocal chunk_index
            if not current_paragraphs:
                return
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    id=f"{source}::{chunk_index}",
                    text=chunk_text,
                    source=source,
                    metadata={"chunk_index": chunk_index},
                )
            )
            chunk_index += 1

        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            para_tokens = _num_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_paragraphs:
                flush()
                # carry over the last paragraph(s) as overlap, trimmed to chunk_overlap tokens
                overlap_paragraphs: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_paragraphs):
                    t = _num_tokens(p)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap_paragraphs.insert(0, p)
                    overlap_tokens += t
                current_paragraphs = overlap_paragraphs
                current_tokens = overlap_tokens
                continue  # re-process the same paragraph against the reset window

            current_paragraphs.append(para)
            current_tokens += para_tokens
            i += 1

        flush()
        return chunks
