"""SPIRE sparse attention mask utilities for Phase 2.

Three composable components (all causal):
  Sink        — always attend to the first sink_size tokens (system prompt + question).
  Local       — always attend to the last local_window tokens (current hop passage + reasoning).
  Hash/Random — randomly select hash_budget tokens from the "middle" (prior accumulated hops).
                In Phase 2 we use random selection as a proxy for LSH hashing; this is
                sufficient to demonstrate the sparse attention effect without key states.
"""

from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt


class SparseAttentionMask:
    """Build SPIRE-style sparse attention masks for causal language models."""

    def __init__(self, sink_size: int = 128, local_window: int = 2048, hash_budget: int = 256):
        """Initialise the three sparse attention hyperparameters.

        Args:
            sink_size:    Number of prefix tokens always attended to.
            local_window: Number of most-recent tokens always attended to.
            hash_budget:  Maximum tokens randomly selected from the "middle" region.
                          Set to 0 to produce the Sink+Local-only ablation (B5).
        """
        self.sink_size = sink_size
        self.local_window = local_window
        self.hash_budget = hash_budget

    # ------------------------------------------------------------------
    # Public mask builders
    # ------------------------------------------------------------------

    def build_mask(
        self,
        seq_len: int,
        key_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a full 2D causal sparse attention mask.

        Used for visualisation and for understanding the policy.
        For efficient KV-cached generation, use build_generation_mask instead.

        Args:
            seq_len:    Length of the sequence.
            key_states: Optional (seq_len, head_dim) float tensor.
                        When provided, selects the hash_budget most similar past tokens
                        via cosine similarity (true semantic sparsity).
                        When None, random selection is used (Phase 2 default).

        Returns:
            Boolean tensor of shape (seq_len, seq_len).
            True  = token i attends to token j.
            False = token i skips token j.
        """
        rows = torch.arange(seq_len).unsqueeze(1)   # (seq_len, 1)
        cols = torch.arange(seq_len).unsqueeze(0)   # (1, seq_len)

        causal = cols <= rows                                              # upper-triangle blocked
        sink   = cols < self.sink_size                                     # first sink_size cols
        local  = (rows - cols) < self.local_window                        # last local_window cols

        mask = causal & (sink | local)

        if self.hash_budget > 0:
            middle = causal & ~sink & ~local          # positions in neither sink nor local window
            for i in range(seq_len):
                candidates = torch.where(middle[i])[0]
                if len(candidates) == 0:
                    continue
                n_select = min(self.hash_budget, len(candidates))
                if key_states is not None and i < key_states.shape[0]:
                    q = key_states[i].float()
                    k = key_states[candidates].float()
                    sims = torch.cosine_similarity(q.unsqueeze(0), k, dim=-1)
                    chosen = sims.topk(n_select).indices
                else:
                    chosen = torch.randperm(len(candidates))[:n_select]
                mask[i, candidates[chosen]] = True

        return mask

    def build_generation_mask(self, total_len: int) -> torch.Tensor:
        """Build a 1D mask for a single KV-cached generation step.

        The new token is at position (total_len - 1).  This specifies which of
        the total_len positions it is allowed to attend to.

        Args:
            total_len: Total sequence length including the new token being generated.

        Returns:
            Long tensor of shape (total_len,): 1 = attend, 0 = skip.
        """
        new_pos = total_len - 1
        mask = torch.zeros(total_len, dtype=torch.long)

        # Sink — always attend to the first sink_size tokens
        mask[: min(self.sink_size, total_len)] = 1

        # Local window — always attend to the last local_window tokens
        local_start = max(0, new_pos - self.local_window + 1)
        mask[local_start:total_len] = 1

        # Hash / random selection from the "middle" between sink and local
        middle_start = min(self.sink_size, local_start)
        middle_end   = local_start
        if middle_end > middle_start and self.hash_budget > 0:
            candidates = torch.arange(middle_start, middle_end)
            n_select   = min(self.hash_budget, len(candidates))
            perm       = torch.randperm(len(candidates))[:n_select]
            mask[candidates[perm]] = 1

        return mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sparsity(self, seq_len: int) -> float:
        """Return the fraction of (causal) attention entries that are skipped."""
        mask  = self.build_mask(seq_len).float()
        total = seq_len * (seq_len + 1) / 2          # causal entries only
        return float(1.0 - mask.tril().sum() / total)

    def visualize(self, seq_len: int, save_path: Optional[Path] = None) -> None:
        """Plot and optionally save the sparse attention pattern.

        Args:
            seq_len:   Number of tokens to visualise.
            save_path: If provided, saves the figure to this path instead of showing it.
        """
        mask_np = self.build_mask(seq_len).float().numpy()
        sparsity_pct = self.sparsity(seq_len) * 100

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(mask_np, aspect="auto", cmap="Blues", origin="upper", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        ax.set_xlabel("Key position (attend to →)")
        ax.set_ylabel("Query position (attending from ↓)")
        ax.set_title(
            f"SPIRE Sparse Attention  |  seq_len={seq_len}\n"
            f"sink={self.sink_size}  local={self.local_window}  hash_budget={self.hash_budget}"
            f"  |  sparsity={sparsity_pct:.1f}%"
        )
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()
