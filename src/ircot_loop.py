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

    def _dedupe_new_passages(self, retrieved_so_far: List[str], new_passages: List[str]) -> List[str]:
        """Return `new_passages` with any texts already present in `retrieved_so_far` removed, preserving order."""
        seen = set(retrieved_so_far)
        unique = []
        for p in new_passages:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return unique

    def _short_query_from_response(self, response: str) -> str:
        """Try to derive a short retrieval query from model response.

        Strategy: take first sentence if short; otherwise ask model to produce
        a one-line query using a deterministic generation call.
        """
        # naive first-sentence extraction
        first_line = response.splitlines()[0].strip()
        # first sentence up to first period
        first_sent = first_line.split(".")[0].strip()
        if 0 < len(first_sent) <= 200 and len(first_sent.split()) <= 30:
            return first_sent

        # fallback: ask model to produce a short query
        system = "You are an assistant that writes concise search queries (one line, <=12 words)."
        user = (
            "From the reasoning below, write a single-line search query (<=12 words) "
            "that would retrieve the most relevant evidence. Do not add explanation.\n\n"
            f"{response}\n\nQuery:"
        )
        try:
            q = self.model.generate([{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=32)
            q = q.strip().splitlines()[0]
            # sanitize: remove trailing punctuation
            return q.rstrip(' .')
        except Exception:
            # last resort: return the original short first line
            return first_sent or response[:200]

    def _force_extract_answer(self, response: str) -> str:
        """When `_extract_answer` fails, ask the model to return the final answer only."""
        system = (
            "You are an assistant that extracts the final answer from model reasoning. "
            "Return the final answer only, no commentary."
        )
        user = f"Here is the model's reasoning:\n\n{response}\n\nReturn final answer only:"
        try:
            out = self.model.generate([{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=64)
            # try to parse with existing extractor first; otherwise return the whole output
            extracted = self._extract_answer(out)
            return extracted if extracted else out.strip()
        except Exception:
            return ""

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
            # dedupe newly retrieved passages against already-seen ones
            new_passages = self._dedupe_new_passages(retrieved_so_far, passages)
            if new_passages:
                retrieved_so_far.extend(new_passages)

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

            # Try to extract explicit answer marker. If absent, do not feed the
            # entire verbose reasoning back into BM25; instead derive a short
            # search query to avoid retrieval drift.
            extracted = self._extract_answer(response)
            if extracted:
                answer = extracted
                break

            # produce a short retrieval query for the next hop (robust to long responses)
            try:
                current_query = self._short_query_from_response(response)
            except Exception:
                current_query = response

        if not answer and reasoning_so_far:
            # If the model never produced the answer marker, ask it to return
            # the final answer only as a concise fallback.
            forced = self._force_extract_answer(reasoning_so_far[-1])
            answer = forced or reasoning_so_far[-1].strip()

        return {
            "question": question,
            "answer": answer,
            "num_hops": len(reasoning_so_far),
            "reasoning_chain": reasoning_so_far,
            "retrieved_passages": retrieved_so_far,
            "context_tokens_per_hop": context_tokens_per_hop,
            "total_tokens": int(sum(context_tokens_per_hop)),
        }
