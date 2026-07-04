"""IRCoT loop implementation for interleaved retrieval-reasoning."""

import re
from typing import Dict, List

from config import SPIREConfig
from src.model_manager import ModelManager
from src.retriever import BM25Retriever


class IRCoTLoop:
    """Run an interleaved retrieval and reasoning loop for one question."""

    def __init__(self, model: ModelManager, retriever: BM25Retriever, config: SPIREConfig):
        """Store model, retriever, and experiment settings."""
        self.model = model
        self.retriever = retriever
        self.config = config

    def _build_messages(
        self,
        question: str,
        retrieved_so_far: List[str],
        reasoning_so_far: List[str],
    ) -> List[Dict[str, str]]:
        """Construct chat-formatted messages for the next IRCoT step."""
        system_prompt = (
            "Answer the question by reasoning step by step. "
            "When you have the final answer, write 'So the answer is: <answer>'."
        )

        evidence_text = "\n\n".join(
            f"[Evidence {idx + 1}] {passage}"
            for idx, passage in enumerate(retrieved_so_far)
        )
        reasoning_text = "\n".join(
            f"Step {idx + 1}: {step}"
            for idx, step in enumerate(reasoning_so_far)
        )

        user_prompt = (
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text if evidence_text else 'None yet.'}\n\n"
            f"Previous reasoning:\n{reasoning_text if reasoning_text else 'None yet.'}\n\n"
            "Continue reasoning step by step."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_answer(self, text: str) -> str | None:
        """Extract final answer from model output if it includes answer marker."""
        match = re.search(
            r"(?:the answer is|answer is)[:\s]*(.+?)(?:\.|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip()

    def run(self, question: str) -> Dict:
        """Run IRCoT loop and return tracing fields for analysis."""
        retrieved_so_far: List[str] = []
        reasoning_so_far: List[str] = []
        context_tokens_per_hop: List[int] = []
        current_query = question
        answer = ""

        for _ in range(self.config.max_hops):
            passages = self.retriever.retrieve(current_query, top_k=self.config.retrieval_top_k)
            retrieved_so_far.extend(passages)

            messages = self._build_messages(question, retrieved_so_far, reasoning_so_far)
            context_length = self.model.get_context_length_from_messages(messages)
            context_tokens_per_hop.append(context_length)

            # Phase 1: dense generation.  Phase 2: sparse generation when use_sparse=True.
            if self.config.use_sparse:
                from src.sparse_attention import SparseAttentionMask
                mask_builder = SparseAttentionMask(
                    sink_size=self.config.sink_size,
                    local_window=self.config.local_window,
                    hash_budget=self.config.hash_budget,
                )
                response = self.model.generate_with_sparse_mask(
                    messages=messages,
                    mask_builder=mask_builder,
                    max_new_tokens=self.config.max_new_tokens,
                )
            else:
                response = self.model.generate(
                    messages=messages,
                    max_new_tokens=self.config.max_new_tokens,
                )
            reasoning_so_far.append(response)

            extracted = self._extract_answer(response)
            if extracted:
                answer = extracted
                break
            current_query = response

        if not answer and reasoning_so_far:
            answer = reasoning_so_far[-1].strip()

        return {
            "question": question,
            "answer": answer,
            "num_hops": len(reasoning_so_far),
            "reasoning_chain": reasoning_so_far,
            "retrieved_passages": retrieved_so_far,
            "context_tokens_per_hop": context_tokens_per_hop,
            "total_tokens": int(sum(context_tokens_per_hop)),
        }
