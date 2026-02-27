"""Precursor m/z index and FAISS embedding index."""

from pathlib import Path

import faiss
import numpy as np


class PrecursorIndex:
    """Sorted-array precursor m/z index with binary search. O(log N) per query."""

    def __init__(self, df):
        self._indices = {}
        for mode in df["mode"].unique():
            sub = df[df["mode"] == mode]
            order = np.argsort(sub["precursor_mz"].values)
            sorted_mz = sub["precursor_mz"].values[order]
            sorted_idx = sub.index.values[order]
            self._indices[mode] = (sorted_mz, sorted_idx)

    def query(self, precursor_mz, mode, tol_da=0.5):
        """Return DataFrame indices within tolerance for the given mode."""
        if mode not in self._indices:
            return np.array([], dtype=np.int64)

        sorted_mz, sorted_idx = self._indices[mode]
        lo = np.searchsorted(sorted_mz, precursor_mz - tol_da, side="left")
        hi = np.searchsorted(sorted_mz, precursor_mz + tol_da, side="right")
        return sorted_idx[lo:hi]


class EmbeddingIndex:
    """FAISS cosine similarity index over L2-normalized embeddings."""

    def __init__(self):
        self._index = None
        self._ids = None

    @property
    def ntotal(self):
        return self._index.ntotal if self._index else 0

    def build(self, embeddings, ids=None):
        """Build index from L2-normalized embeddings."""
        emb = embeddings.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

        dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb)
        self._ids = ids if ids is not None else np.arange(len(emb))

    def query(self, query_embedding, top_k=50):
        """Return [(id, score)] for top-k nearest neighbors."""
        assert self._index is not None, "Index not built"

        q = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        return [
            (int(self._ids[indices[0, i]]), float(scores[0, i]))
            for i in range(k)
            if indices[0, i] >= 0
        ]

    def batch_query(self, query_embeddings, top_k=50):
        """Batch query. Returns list of [(id, score)] per query."""
        assert self._index is not None, "Index not built"

        q = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for i in range(len(q)):
            row = [
                (int(self._ids[indices[i, j]]), float(scores[i, j]))
                for j in range(k)
                if indices[i, j] >= 0
            ]
            results.append(row)
        return results

    def save(self, path):
        """Save index and id mapping to disk."""
        assert self._index is not None
        path = Path(path)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        np.save(str(path.with_suffix(".ids.npy")), self._ids)

    def load(self, path):
        """Load index and id mapping from disk."""
        path = Path(path)
        self._index = faiss.read_index(str(path.with_suffix(".faiss")))
        self._ids = np.load(str(path.with_suffix(".ids.npy")))
