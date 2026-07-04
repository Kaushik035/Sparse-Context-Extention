"""Phase 3: Attention-guided retrieval for the SPIRE IRCoT loop.

After each reasoning step the model has already attended over the entire context.
This module extracts those attention weights and uses them as a retrieval signal:
passages that the reasoning tokens attended to most heavily are the ones the model
"looked for" — they are likely to contain the next needed evidence.

Reference: AttentionRetriever (Fu et al., 2026 · arXiv:2602.12278)
  - Layers in the second half of the network yield the best retrieval accuracy.
  - Scoring: for each reasoning token, take the max attention it gives to any
    token inside the passage, then average over reasoning tokens and layers/heads.

Memory safety:
  Storing full attention maps for a 32-layer 8B model on a long sequence can
  exceed GPU VRAM.  A configurable guard (MAX_SEQ_LEN, default 1500 tokens)
  falls back to BM25 when the input would be too large.
"""

from typing import List, Optional, Tuple

import torch

from src.model_manager import ModelManager
from src.retriever import BM25Retriever

# Attention retrieval is disabled if the tokenised input exceeds this length
# to prevent OOM on lower-spec hardware.  Increase on A100 if needed.
MAX_ATTENTION_SEQ_LEN: int = 1500


class AttentionRetriever:
    """Retrieve passages by scoring them with the model's own attention maps.

    At each IRCoT hop, instead of using BM25 on the generated reasoning text,
    this retriever runs a single forward pass over
        [SYS + Question] [Passage_1] [Passage_2] ... [Passage_N] [Reasoning]
    and scores each passage by how strongly the reasoning tokens attended to it.
    """

    def __init__(
        self,
        model: ModelManager,
        passages: List[str],
        bm25_retriever: BM25Retriever,
        max_seq_len: int = MAX_ATTENTION_SEQ_LEN,
    ):
        """Initialise with model, candidate passage pool, and a BM25 fallback.

        Args:
            model:           Shared ModelManager instance (tokenizer + HF model).
            passages:        Candidate passages for this example (same pool as BM25).
            bm25_retriever:  BM25Retriever over the same passage pool — used as
                             fallback when the sequence is too long for attention.
            max_seq_len:     Maximum total token length before falling back to BM25.
        """
        self.model = model
        self.passages = passages
        self.bm25 = bm25_retriever
        self.max_seq_len = max_seq_len

        # Use the second half of the network as target layers (highest retrieval
        # accuracy per AttentionRetriever paper; applied to both 1B and 8B models).
        num_layers = model.model.config.num_hidden_layers
        half = num_layers // 2
        self.target_layer_indices = set(range(half, num_layers))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        reasoning_so_far: List[str],
        top_k: int = 3,
    ) -> List[str]:
        """Return top-k passages ranked by attention from the model's reasoning.

        Falls back to BM25 (using the last reasoning step as query) when:
          - reasoning_so_far is empty (hop 0 — no reasoning yet), or
          - the tokenised input would exceed max_seq_len.

        Args:
            question:          The original multi-hop question.
            reasoning_so_far:  List of reasoning steps generated so far.
            top_k:             Number of passages to return.

        Returns:
            List of at most top_k passage strings, ordered by relevance.
        """
        # --- Hop 0: no reasoning generated yet — fall back to BM25 on question ---
        if not reasoning_so_far:
            return self.bm25.retrieve(question, top_k=top_k)

        # --- Build tokenised input and check length ---
        input_ids, passage_ranges, reasoning_range = self._build_input(
            question=question,
            reasoning_text=" ".join(reasoning_so_far),
        )

        if input_ids.shape[-1] > self.max_seq_len:
            fallback_query = reasoning_so_far[-1]
            return self.bm25.retrieve(fallback_query, top_k=top_k)

        # --- Forward pass with attention output ---
        device = self.model.model.device
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids.to(device),
                output_attentions=True,
            )

        # --- Score passages, free memory immediately ---
        scores = self._score_passages(
            attentions=outputs.attentions,
            passage_ranges=passage_ranges,
            reasoning_range=reasoning_range,
        )
        del outputs  # free attention tensors

        # --- Return top-k by descending score ---
        ranked = sorted(range(len(self.passages)), key=lambda i: scores[i], reverse=True)
        return [self.passages[i] for i in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input(
        self,
        question: str,
        reasoning_text: str,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], Tuple[int, int]]:
        """Tokenise [prefix | passage_0 | ... | passage_N | reasoning] and
        return the input tensor plus token-position ranges for each passage
        and the reasoning block.

        Args:
            question:       The multi-hop question.
            reasoning_text: All reasoning steps joined into one string.

        Returns:
            input_ids:       (1, total_len) long tensor.
            passage_ranges:  List of (start, end) token indices, one per passage.
            reasoning_range: (start, end) token indices for the reasoning block.
        """
        tok = self.model.tokenizer

        # Prefix: question anchor (always attended — mirrors the sink policy)
        prefix_ids = tok.encode(
            f"Question: {question}\n\n",
            add_special_tokens=True,
        )
        all_ids: List[int] = list(prefix_ids)
        passage_ranges: List[Tuple[int, int]] = []

        for passage in self.passages:
            p_ids = tok.encode(f"\n{passage}\n", add_special_tokens=False)
            start = len(all_ids)
            all_ids.extend(p_ids)
            passage_ranges.append((start, len(all_ids)))

        # Reasoning block (the "query" whose attention we analyse)
        r_ids = tok.encode(f"\nReasoning: {reasoning_text}", add_special_tokens=False)
        reasoning_start = len(all_ids)
        all_ids.extend(r_ids)
        reasoning_range = (reasoning_start, len(all_ids))

        input_ids = torch.tensor([all_ids], dtype=torch.long)
        return input_ids, passage_ranges, reasoning_range

    def _score_passages(
        self,
        attentions: tuple,
        passage_ranges: List[Tuple[int, int]],
        reasoning_range: Tuple[int, int],
    ) -> List[float]:
        """Score each passage by the attention reasoning tokens give it.

        Scoring formula (per AttentionRetriever):
          score(passage_i) = mean over target_layers of
                             mean over heads of
                             mean over reasoning_tokens of
                             max over passage_tokens of
                             attention[layer][head][reasoning_token, passage_token]

        Args:
            attentions:      Tuple of (batch, heads, seq, seq) tensors, one per layer.
            passage_ranges:  (start, end) token ranges for each passage.
            reasoning_range: (start, end) token range for the reasoning block.

        Returns:
            Float score per passage (higher = more relevant).
        """
        r_start, r_end = reasoning_range
        scores = [0.0] * len(passage_ranges)
        n_contributing_layers = 0

        for layer_idx, layer_attn in enumerate(attentions):
            if layer_idx not in self.target_layer_indices:
                continue

            # layer_attn: (1, heads, seq, seq) → (heads, seq, seq)
            layer_attn_cpu = layer_attn[0].cpu().float()
            n_contributing_layers += 1

            for p_idx, (p_start, p_end) in enumerate(passage_ranges):
                if p_start >= p_end or p_end > layer_attn_cpu.shape[-1]:
                    continue
                if r_start >= r_end or r_end > layer_attn_cpu.shape[-2]:
                    continue

                # attn_slice: (heads, r_len, p_len)
                attn_slice = layer_attn_cpu[:, r_start:r_end, p_start:p_end]
                if attn_slice.numel() == 0:
                    continue

                # max over passage tokens → (heads, r_len), then mean over all
                passage_score = float(attn_slice.max(dim=-1).values.mean())
                scores[p_idx] += passage_score

        if n_contributing_layers > 0:
            scores = [s / n_contributing_layers for s in scores]

        return scores
