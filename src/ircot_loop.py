"""IRCoT loop implementation for interleaved retrieval-reasoning."""

import re
from typing import Dict, List

from config import SPIREConfig
from src.model_manager import ModelManager
from src.retriever import BM25Retriever


class IRCoTLoop:
    """Run an interleaved retrieval and reasoning loop for one question."""

    def __init__(
        self,
        model: ModelManager,
        retriever: BM25Retriever,
        config: SPIREConfig,
        attention_retriever=None,
    ):
        """Store model, retriever, and experiment settings.

        Args:
            model:               Shared ModelManager.
            retriever:           BM25Retriever for this example's passage pool.
            config:              SPIREConfig with all phase flags.
            attention_retriever: Optional AttentionRetriever (Phase 3).
                                 When provided and config.use_attention_retrieval=True,
                                 replaces BM25 as the retrieval signal after hop 0.
        """
        self.model = model
        self.retriever = retriever
        self.config = config
        self.attention_retriever = attention_retriever

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

        # B3 — IRCoT-Truncate: keep only the last truncate_context_tokens tokens of
        # accumulated evidence + reasoning.  This is the naïve compression baseline.
        if self.config.truncate_context_tokens > 0 and (evidence_text or reasoning_text):
            combined = (
                f"Evidence:\n{evidence_text}\n\nPrevious reasoning:\n{reasoning_text}"
            )
            token_ids = self.model.tokenizer.encode(
                combined, add_special_tokens=False
            )
            if len(token_ids) > self.config.truncate_context_tokens:
                token_ids = token_ids[-self.config.truncate_context_tokens :]
                combined = self.model.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )
            user_prompt = (
                f"Question: {question}\n\n"
                f"{combined}\n\n"
                "Continue reasoning step by step."
            )
        else:
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
            # --- Retrieval ---
            # Phase 3: attention-guided retrieval replaces BM25 after the first hop.
            # Phase 1 & 2: always use BM25.
            if (
                self.config.use_attention_retrieval
                and self.attention_retriever is not None
            ):
                passages = self.attention_retriever.retrieve(
                    question=question,
                    reasoning_so_far=reasoning_so_far,
                    top_k=self.config.retrieval_top_k,
                )
            else:
                passages = self.retriever.retrieve(
                    current_query, top_k=self.config.retrieval_top_k
                )
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
