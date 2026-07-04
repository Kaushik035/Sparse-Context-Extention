"""Model loading and text generation utilities for SPIRE."""

import os
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SPIREConfig


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
