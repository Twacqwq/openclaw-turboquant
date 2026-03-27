"""Context compression engine for OpenClaw integration.

This module provides a high-level API for compressing and retrieving
context vectors using TurboQuant. It is designed to back an OpenClaw
Context Engine plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from openclaw_turboquant.quantizer import ProdQuantized, TurboQuantProd


@dataclass(slots=True)
class ContextEntry:
    """A single stored context entry with quantized embedding."""

    entry_id: str
    text: str
    quantized: ProdQuantized
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextStore:
    """In-memory store of quantized context vectors with similarity search.

    Uses TurboQuantProd for unbiased inner-product estimation, enabling
    fast approximate nearest-neighbor retrieval on compressed vectors.

    Parameters
    ----------
    d : int
        Embedding dimension.
    bit_width : int
        Bits per coordinate for quantization.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        d: int,
        bit_width: int = 4,
        *,
        seed: int | None = None,
    ) -> None:
        self.d = d
        self.bit_width = bit_width
        self.quantizer = TurboQuantProd(d, bit_width, seed=seed)
        self._entries: dict[str, ContextEntry] = {}

    @property
    def size(self) -> int:
        """Number of stored entries."""
        return len(self._entries)

    def ingest(
        self,
        entry_id: str,
        embedding: NDArray[np.float64],
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a context entry.

        Parameters
        ----------
        entry_id : str
            Unique identifier for this entry.
        embedding : NDArray[np.float64]
            Embedding vector of shape (d,).
        text : str
            Raw text content.
        metadata : dict or None
            Optional metadata.
        """
        quantized = self.quantizer.quantize(embedding)
        self._entries[entry_id] = ContextEntry(
            entry_id=entry_id,
            text=text,
            quantized=quantized,
            metadata=metadata or {},
        )

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        return self._entries.pop(entry_id, None) is not None

    def retrieve_top_k(
        self,
        query_embedding: NDArray[np.float64],
        k: int = 5,
    ) -> list[tuple[ContextEntry, float]]:
        """Retrieve the top-k most similar entries by inner product.

        Parameters
        ----------
        query_embedding : NDArray[np.float64]
            Query vector of shape (d,).
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[ContextEntry, float]]
            Entries with their estimated similarity scores, sorted descending.
        """
        if not self._entries:
            return []

        scores: list[tuple[ContextEntry, float]] = []
        for entry in self._entries.values():
            score = self.quantizer.estimate_inner_product(query_embedding, entry.quantized)
            scores.append((entry, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def assemble_context(
        self,
        query_embedding: NDArray[np.float64],
        token_budget: int,
        *,
        avg_chars_per_token: float = 4.0,
    ) -> list[dict[str, Any]]:
        """Build context messages that fit within a token budget.

        Retrieves entries by relevance and greedily fills the budget.

        Parameters
        ----------
        query_embedding : NDArray[np.float64]
            Query embedding for relevance ranking.
        token_budget : int
            Max tokens to use.
        avg_chars_per_token : float
            Approximate characters per token for budgeting.

        Returns
        -------
        list[dict[str, Any]]
            Context messages in OpenClaw-compatible format.
        """
        char_budget = int(token_budget * avg_chars_per_token)
        ranked = self.retrieve_top_k(query_embedding, k=self.size)

        messages: list[dict[str, Any]] = []
        used_chars = 0

        for entry, score in ranked:
            entry_chars = len(entry.text)
            if used_chars + entry_chars > char_budget:
                continue
            messages.append(
                {
                    "role": "context",
                    "content": entry.text,
                    "metadata": {
                        "entry_id": entry.entry_id,
                        "relevance_score": score,
                        **entry.metadata,
                    },
                }
            )
            used_chars += entry_chars

        return messages

    def compact(
        self,
        keep_ratio: float = 0.5,
        query_embedding: NDArray[np.float64] | None = None,
    ) -> int:
        """Compact the store by removing least relevant entries.

        Parameters
        ----------
        keep_ratio : float
            Fraction of entries to keep (0.0–1.0).
        query_embedding : NDArray or None
            If provided, keep entries most similar to this query.
            If None, keep the most recently added entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        if not self._entries or keep_ratio >= 1.0:
            return 0

        keep_count = max(1, int(self.size * keep_ratio))

        if query_embedding is not None:
            ranked = self.retrieve_top_k(query_embedding, k=self.size)
            keep_ids = {entry.entry_id for entry, _ in ranked[:keep_count]}
        else:
            # Keep last N entries (insertion order)
            all_ids = list(self._entries.keys())
            keep_ids = set(all_ids[-keep_count:])

        remove_ids = [eid for eid in self._entries if eid not in keep_ids]
        for eid in remove_ids:
            del self._entries[eid]

        return len(remove_ids)

    def memory_estimate_bytes(self) -> int:
        """Estimate memory usage of the compressed store.

        Returns
        -------
        int
            Approximate bytes used by quantized data.
        """
        if not self._entries:
            return 0

        # Per entry: (b-1)*d bits for MSE indices + d bits for QJL signs + 64 bits for norms
        bits_per_entry = (self.bit_width - 1) * self.d + self.d + 128
        return (bits_per_entry * self.size + 7) // 8
