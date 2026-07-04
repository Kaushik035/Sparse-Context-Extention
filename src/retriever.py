"""BM25 retriever utilities for SPIRE."""

from typing import List

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Simple BM25 retriever over per-example passage pools."""

    def __init__(self, passages: List[str]):
        """Index passages with BM25."""
        self.passages = [p for p in passages if p and p.strip()]
        tokenized = [self._tokenize(p) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase whitespace tokens for BM25."""
        return text.lower().split()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k passages for query."""
        if not self.passages:
            return []

        query_tokens = self._tokenize(query)
        ranked = self.bm25.get_top_n(query_tokens, self.passages, n=min(top_k, len(self.passages)))
        return ranked
