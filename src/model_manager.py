"""Model loading and text generation utilities for SPIRE."""

import os
from typing import TYPE_CHECKING, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SPIREConfig

if TYPE_CHECKING:
    from src.sparse_attention import SparseAttentionMask


class ModelManager:
    """Manage tokenizer/model lifecycle and generation calls."""

    def __init__(self, config: SPIREConfig):
        """Initialize model + tokenizer using Hugging Face credentials from env."""
        self.config = config
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise EnvironmentError("HF_TOKEN is missing. Add it to .env before running.")

        dtype = self._resolve_dtype(config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=self.hf_token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            token=self.hf_token,
            dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        self.generated_token_counts: List[int] = []

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        """Map string dtype name from config into a torch dtype."""
        normalized = dtype_name.lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
        return mapping[normalized]

    def _prepare_inputs(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Tokenize chat messages with the model chat template."""
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Newer transformers versions return a BatchEncoding; extract the tensor.
        input_ids = result.input_ids if hasattr(result, "input_ids") else result
        return input_ids.to(self.model.device)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        """Generate response from chat messages and return generated text only."""
        input_ids = self._prepare_inputs(messages)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0, input_ids.shape[-1] :]
        self.generated_token_counts.append(int(completion_ids.shape[-1]))
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens in plain text with the model tokenizer."""
        return int(len(self.tokenizer.encode(text, add_special_tokens=False)))

    def get_context_length(self, prompt: str) -> int:
        """Return tokenized length of a plain text prompt."""
        return self.count_tokens(prompt)

    def get_context_length_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """Return token length for chat-formatted messages."""
        input_ids = self._prepare_inputs(messages)
        return int(input_ids.shape[-1])

    # ------------------------------------------------------------------
    # Phase 2 — sparse-attention generation
    # ------------------------------------------------------------------

    def generate_with_sparse_mask(
        self,
        messages: List[Dict[str, str]],
        mask_builder: "SparseAttentionMask",
        max_new_tokens: int = 256,
    ) -> str:
        """Generate with SPIRE sparse attention applied at every generation step.

        Strategy:
          1. Run a full dense prefill over the prompt to populate the KV cache.
          2. Generate each subsequent token using a 1D sparse attention mask that
             controls which past KV entries the model may attend to (sink + local
             window + random-hash selection from old cycles).

        This keeps all context tokens accessible (no information loss vs truncation)
        but concentrates the model's attention budget on fresh evidence and the
        original question — exactly the SPIRE policy described in the research proposal.

        Args:
            messages:       Chat-formatted messages (same format as generate()).
            mask_builder:   SparseAttentionMask instance with configured parameters.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text only (prompt stripped).
        """
        input_ids = self._prepare_inputs(messages)
        input_len = int(input_ids.shape[-1])
        device    = input_ids.device

        # ---- Step 1: dense prefill — build KV cache ----
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                use_cache=True,
            )

        past_key_values = prefill_out.past_key_values
        next_logits     = prefill_out.logits[:, -1, :]   # logits at the last prompt position

        generated_ids: List[int] = []

        # ---- Step 2: sparse generation loop ----
        for step in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1, keepdim=True)   # (1, 1)
            generated_ids.append(int(next_token.item()))

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # total_len covers: all prompt positions + tokens generated so far.
            # The new token we just appended is at position (input_len + step).
            # For the NEXT step the KV cache will contain input_len + step + 1 entries,
            # so the attention_mask needs to cover exactly that many positions.
            total_len = input_len + step + 1
            sparse_1d  = mask_builder.build_generation_mask(total_len)       # (total_len,)
            attn_mask  = sparse_1d.unsqueeze(0).to(device)                   # (1, total_len)

            with torch.no_grad():
                step_out = self.model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = step_out.past_key_values
            next_logits     = step_out.logits[:, -1, :]

        self.generated_token_counts.append(len(generated_ids))
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
