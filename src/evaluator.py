"""Evaluation utilities (F1/EM) for SPIRE experiments."""

import re
import string
from collections import defaultdict
from typing import Dict, List, Optional


class Evaluator:
    """Compute normalized Exact Match and token-level F1."""

    def __init__(self):
        """Initialize evaluator state."""

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lowercase, remove punctuation/articles, and fix whitespace."""
        text = s.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 between normalized strings."""
        pred_tokens = Evaluator.normalize_answer(prediction).split()
        gold_tokens = Evaluator.normalize_answer(ground_truth).split()

        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        pred_counts = {}
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1

        gold_counts = {}
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1

        overlap = 0
        for token, count in pred_counts.items():
            if token in gold_counts:
                overlap += min(count, gold_counts[token])

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Check exact match after normalization."""
        return Evaluator.normalize_answer(prediction) == Evaluator.normalize_answer(ground_truth)

    def evaluate_results(
        self,
        results: List[Dict],
        gold_answers: List[str],
        hop_depths: Optional[List[int]] = None,
    ) -> Dict:
        """Compute aggregate metrics and hop-wise slices."""
        if len(results) != len(gold_answers):
            raise ValueError("results and gold_answers must have the same length")
        if hop_depths is not None and len(hop_depths) != len(results):
            raise ValueError("hop_depths must match results length when provided")

        f1_scores: List[float] = []
        em_scores: List[float] = []

        f1_by_hops_raw: Dict[int, List[float]] = defaultdict(list)
        context_tokens_by_hop: Dict[int, List[int]] = defaultdict(list)

        for idx, result in enumerate(results):
            pred = result.get("answer", "")
            gold = gold_answers[idx]

            f1 = self.f1_score(pred, gold)
            em = 1.0 if self.exact_match(pred, gold) else 0.0
            f1_scores.append(f1)
            em_scores.append(em)

            hop_bucket = hop_depths[idx] if hop_depths is not None else int(result.get("num_hops", 0))
            f1_by_hops_raw[hop_bucket].append(f1)

            per_hop_context = result.get("context_tokens_per_hop", [])
            for hop_index, token_count in enumerate(per_hop_context, start=1):
                context_tokens_by_hop[hop_index].append(int(token_count))

        f1_by_hops = {
            hop: (sum(values) / len(values) if values else 0.0)
            for hop, values in sorted(f1_by_hops_raw.items())
        }

        return {
            "overall_f1": (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
            "overall_em": (sum(em_scores) / len(em_scores)) if em_scores else 0.0,
            "f1_by_hops": f1_by_hops,
            "context_tokens_by_hop": {k: v for k, v in sorted(context_tokens_by_hop.items())},
        }
