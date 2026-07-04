"""Dense and hybrid retrieval for SPIRE — Baselines B8 and B9.

B8 — CosineRetriever
    Encodes all passages once with a sentence-transformer, encodes the query
    at retrieval time, and returns top-k by cosine similarity.  This is a
    semantic-similarity baseline that goes beyond BM25's keyword overlap.

B9 — HybridRetriever
    Combines BM25 and cosine scores with configurable weighting (default 50/50).
    Both score lists are min-max normalised before mixing so they live on [0, 1].
    This tests whether the two signals are complementary — the expected outcome
    is that hybrid >= cosine >= BM25 on average when BM25 and cosine agree, and
    hybrid degrades gracefully when one signal is noisy.

Design notes
────────────
* Passage embeddings are computed once during __init__ (not per query) so that
  repeated calls to retrieve() inside the IRCoT loop pay only a single matrix
  multiply overhead, not an N-passage encoding overhead.
* The class interface matches BM25Retriever exactly (retrieve(query, top_k))
  so the IRCoT loop and Phase 3 runner can use either without changes.
* sentence-transformers is imported lazily so that Phase 1 / Phase 2 runs that
  do not need it are not impacted even if the package is not yet installed.
"""

from __future__ import annotations

from typing import List

import torch


class CosineRetriever:
    """Retrieve passages by cosine similarity to a sentence-transformer embedding.

    Passage embeddings are pre-computed at construction time and cached on CPU.
    Query encoding runs at retrieval time (one forward pass per hop).

    Args:
        passages:    List of passage strings.  Filtered to non-empty on init.
        model_name:  Sentence-transformer model to use for encoding.
                     Default: "sentence-transformers/all-MiniLM-L6-v2"
                     (fast, 384-dim, good retrieval quality for most tasks).
    """

    def __init__(
        self,
        passages: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        st_model=None,  # pre-built SentenceTransformer; avoids reloading per example
    ) -> None:
        self.passages = [p for p in passages if p and p.strip()]
        if not self.passages:
            self._passage_embeddings = None
            self._model = None
            return

        if st_model is not None:
            self._model = st_model
        else:
            # Lazy import so callers without sentence-transformers installed still
            # import dense_retriever without raising ImportError at module load time.
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for B8/B9. "
                    "Run: pip install sentence-transformers>=2.7"
                ) from exc
            self._model = SentenceTransformer(model_name)

        # Encode all passages once; normalise to unit norm for fast cosine via dot product.
        raw = self._model.encode(
            self.passages,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # (N, dim)
        # Keep on CPU to avoid VRAM pressure during model inference.
        self._passage_embeddings = raw.float().cpu()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return top-k passages by cosine similarity to query.

        Args:
            query:  Retrieval query string (reasoning step or question).
            top_k:  Maximum number of passages to return.

        Returns:
            Ordered list of at most top-k passage strings (highest-score first).
        """
        if not self.passages or self._passage_embeddings is None or self._model is None:
            return []

        # Encode query (normalised so dot-product = cosine similarity).
        q_emb = self._model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float().cpu()  # (1, dim)

        # Cosine scores via dot product (both sides are L2-normalised).
        scores = (self._passage_embeddings @ q_emb.T).squeeze(1)  # (N,)

        k = min(top_k, len(self.passages))
        top_indices = scores.topk(k).indices.tolist()
        return [self.passages[i] for i in top_indices]


class HybridRetriever:
    """Retrieve passages by combining BM25 and cosine scores.

    Both raw score lists are independently min-max normalised to [0, 1] before
    being mixed.  The final score for passage i is:

        score_i = w * cosine_norm_i + (1 - w) * bm25_norm_i

    where w = dense_weight (default 0.5).

    Args:
        passages:       List of passage strings.
        bm25_retriever: Pre-built BM25Retriever over the same passage pool.
        model_name:     Sentence-transformer model for the cosine side.
        dense_weight:   Weight for the cosine component (0 = pure BM25,
                        1 = pure cosine, 0.5 = equal mix).
    """

    def __init__(
        self,
        passages: List[str],
        bm25_retriever,           # BM25Retriever — avoid circular import
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.5,
        st_model=None,  # pre-built SentenceTransformer; passed through to CosineRetriever
    ) -> None:
        self.passages = [p for p in passages if p and p.strip()]
        self.bm25 = bm25_retriever
        self.dense_weight = float(dense_weight)

        # Build the cosine side (embeddings cached at init).
        self._cosine = CosineRetriever(passages=self.passages, model_name=model_name, st_model=st_model)

    def _bm25_scores_all(self, query: str) -> List[float]:
        """Return raw BM25 score for every passage in the pool."""
        query_tokens = query.lower().split()
        scores = self.bm25.bm25.get_scores(query_tokens).tolist()
        return scores

    def _cosine_scores_all(self, query: str) -> List[float]:
        """Return cosine similarity score for every passage in the pool."""
        if not self._cosine.passages or self._cosine._passage_embeddings is None:
            return [0.0] * len(self.passages)

        try:
            from sentence_transformers import SentenceTransformer as _ST  # noqa: F401
        except ImportError:
            return [0.0] * len(self.passages)

        q_emb = self._cosine._model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float().cpu()
        scores = (self._cosine._passage_embeddings @ q_emb.T).squeeze(1)
        return scores.tolist()

    @staticmethod
    def _minmax_normalize(scores: List[float]) -> List[float]:
        """Min-max normalise a score list to [0, 1]; return uniform 0.5 if flat."""
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return top-k passages by hybrid BM25 + cosine score.

        Args:
            query:  Retrieval query string.
            top_k:  Maximum number of passages to return.

        Returns:
            Ordered list of at most top-k passage strings (highest-score first).
        """
        if not self.passages:
            return []

        bm25_raw = self._bm25_scores_all(query)
        cosine_raw = self._cosine_scores_all(query)

        # Normalise independently.
        bm25_norm = self._minmax_normalize(bm25_raw)
        cosine_norm = self._minmax_normalize(cosine_raw)

        w = self.dense_weight
        fused = [
            w * cosine_norm[i] + (1.0 - w) * bm25_norm[i]
            for i in range(len(self.passages))
        ]

        k = min(top_k, len(self.passages))
        ranked = sorted(range(len(self.passages)), key=lambda i: fused[i], reverse=True)
        return [self.passages[i] for i in ranked[:k]]
