"""Phase 2: Compare Dense IRCoT vs SPIRE sparse variants on MuSiQue.

Baselines run in this script
─────────────────────────────
B2  dense       — IRCoT with full attention (same as Phase 1).
                  Reloaded from a Phase 1 JSON artifact if one exists,
                  otherwise re-run fresh.
B5  spire_b5    — SPIRE Sink + Local only (no hash / random selection).
                  hash_budget=0 disables the middle-region selection.
B6  spire_b6    — SPIRE Full: Sink + Local + random hash selection.

Outputs (results/phase2/)
─────────────────────────
run_<timestamp>.json         — per-example results + metrics for all three configs
mask_pattern_<timestamp>.png — SPIRE sparse attention pattern visualisation
f1_by_hop_<timestamp>.png    — F1 vs hop depth for all three configs (comparison)
context_tokens_<timestamp>.png — context token growth (same across configs, for reference)
memory_<timestamp>.png       — GPU memory at each hop (if CUDA available)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import SPIREConfig
from src.evaluator import Evaluator
from src.ircot_loop import IRCoTLoop
from src.model_manager import ModelManager
from src.retriever import BM25Retriever
from src.sparse_attention import SparseAttentionMask


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure root logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Dataset loading (reuses run_phase1 helpers via import)
# ---------------------------------------------------------------------------

def load_musique_examples(config: SPIREConfig) -> Tuple[List[Dict], List[str], List[int]]:
    """Load, filter, and slice MuSiQue examples — identical logic to Phase 1."""
    # Import helpers from run_phase1 to avoid duplication
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import (
        load_musique_validation,
        filter_answerable,
        get_passages,
        infer_hop_depth,
    )

    examples = load_musique_validation()
    examples = filter_answerable(examples)
    examples = examples[: config.num_examples]

    gold_answers: List[str] = [ex.get("answer", "") for ex in examples]
    hop_depths:   List[int] = [infer_hop_depth(ex) for ex in examples]

    return examples, gold_answers, hop_depths


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def gpu_memory_gb() -> float:
    """Return currently allocated GPU memory in GB, or 0 if no CUDA device."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Single-config run
# ---------------------------------------------------------------------------

def run_one_config(
    config: SPIREConfig,
    model_manager: ModelManager,
    examples: List[Dict],
    label: str,
) -> Tuple[List[Dict], List[float]]:
    """Run IRCoT on all examples for a given config.  Returns (results, memory_per_hop).

    memory_per_hop[k] = average GPU memory (GB) when generating at hop k+1.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import get_passages

    results: List[Dict] = []
    # Accumulate GPU memory readings indexed by generation hop (1-based)
    memory_by_hop: Dict[int, List[float]] = defaultdict(list)

    for example in tqdm(examples, desc=f"Phase 2 [{label}]"):
        question = example.get("question", "")
        passages = get_passages(example)
        if not question or not passages:
            continue

        retriever = BM25Retriever(passages=passages)
        loop = IRCoTLoop(model=model_manager, retriever=retriever, config=config)

        # Wrap the loop's run to capture per-hop GPU memory
        result = _run_with_memory_tracking(loop, question, memory_by_hop)
        results.append(result)

    # Average memory per hop
    avg_memory = [
        float(sum(memory_by_hop[h]) / len(memory_by_hop[h]))
        for h in sorted(memory_by_hop.keys())
    ]
    return results, avg_memory


def _run_with_memory_tracking(
    loop: IRCoTLoop,
    question: str,
    memory_by_hop: Dict[int, List[float]],
) -> Dict:
    """Run the IRCoT loop and record GPU memory usage at each hop."""
    from src.sparse_attention import SparseAttentionMask

    config = loop.config
    retrieved_so_far: List[str] = []
    reasoning_so_far: List[str] = []
    context_tokens_per_hop: List[int] = []
    current_query = question
    answer = ""

    for hop_idx in range(config.max_hops):
        passages = loop.retriever.retrieve(current_query, top_k=config.retrieval_top_k)
        retrieved_so_far.extend(passages)

        messages = loop._build_messages(question, retrieved_so_far, reasoning_so_far)
        context_length = loop.model.get_context_length_from_messages(messages)
        context_tokens_per_hop.append(context_length)

        mem_before = gpu_memory_gb()

        if config.use_sparse:
            mask_builder = SparseAttentionMask(
                sink_size=config.sink_size,
                local_window=config.local_window,
                hash_budget=config.hash_budget,
            )
            response = loop.model.generate_with_sparse_mask(
                messages=messages,
                mask_builder=mask_builder,
                max_new_tokens=config.max_new_tokens,
            )
        else:
            response = loop.model.generate(
                messages=messages,
                max_new_tokens=config.max_new_tokens,
            )

        mem_after = gpu_memory_gb()
        memory_by_hop[hop_idx + 1].append(max(mem_before, mem_after))

        reasoning_so_far.append(response)

        extracted = loop._extract_answer(response)
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


# ---------------------------------------------------------------------------
# Load Phase 1 dense results (if available)
# ---------------------------------------------------------------------------

def try_load_phase1_dense(phase1_dir: Path) -> Optional[Tuple[List[Dict], List[str], List[int]]]:
    """Attempt to load the most recent Phase 1 run artifact for the dense baseline.

    Returns (results, gold_answers, hop_depths) on success, None if not found.
    """
    candidates = sorted(phase1_dir.glob("run_*.json"), reverse=True)
    if not candidates:
        return None

    path = candidates[0]
    logging.info("Loading dense baseline from Phase 1 artifact: %s", path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    results     = payload.get("results", [])
    gold_answers = payload.get("gold_answers", [])
    hop_depths   = payload.get("hop_depths", [])

    if results and gold_answers:
        return results, gold_answers, hop_depths
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_f1_comparison(
    metrics_map: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Plot F1 vs hop depth for all configurations on one figure."""
    styles = {
        "dense":    ("o", "#1f77b4", "-",  "B2 Dense IRCoT"),
        "spire_b5": ("s", "#ff7f0e", "--", "B5 SPIRE Sink+Local"),
        "spire_b6": ("^", "#2ca02c", "-.", "B6 SPIRE Full"),
    }

    plt.figure(figsize=(9, 5))
    for key, metrics in metrics_map.items():
        f1_by_hops = metrics.get("f1_by_hops", {})
        if not f1_by_hops:
            continue
        hops   = sorted(int(k) for k in f1_by_hops.keys())
        values = [float(f1_by_hops[h]) for h in hops]
        marker, color, ls, label = styles.get(key, ("o", "gray", "-", key))
        plt.plot(hops, values, marker=marker, linestyle=ls, color=color, linewidth=2, label=label)

    plt.xlabel("Hop Depth")
    plt.ylabel("F1")
    plt.title("Phase 2: F1 vs Hop Depth — Dense IRCoT vs SPIRE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory_comparison(
    memory_map: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """Plot GPU memory usage at each hop for all configurations."""
    if not any(memory_map.values()):
        logging.info("No GPU memory data (CPU run?) — skipping memory plot.")
        return

    styles = {
        "dense":    ("#1f77b4", "-",  "B2 Dense IRCoT"),
        "spire_b5": ("#ff7f0e", "--", "B5 SPIRE Sink+Local"),
        "spire_b6": ("#2ca02c", "-.", "B6 SPIRE Full"),
    }

    plt.figure(figsize=(9, 5))
    for key, mem_values in memory_map.items():
        if not mem_values:
            continue
        hops = list(range(1, len(mem_values) + 1))
        color, ls, label = styles.get(key, ("gray", "-", key))
        plt.plot(hops, mem_values, linestyle=ls, color=color, linewidth=2, label=label)

    plt.xlabel("IRCoT Generation Hop")
    plt.ylabel("GPU Memory Allocated (GB)")
    plt.title("Phase 2: GPU Memory per Hop")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_context_tokens(metrics_map: Dict[str, Dict], output_path: Path) -> None:
    """Plot average context token count per hop (same across configs — sanity check)."""
    # Use any config's data (context growth is config-independent)
    for metrics in metrics_map.values():
        tokens_by_hop = metrics.get("context_tokens_by_hop", {})
        if not tokens_by_hop:
            continue
        hops = sorted(int(k) for k in tokens_by_hop.keys())
        avgs = [sum(tokens_by_hop[h]) / len(tokens_by_hop[h]) for h in hops]
        plt.figure(figsize=(9, 5))
        plt.plot(hops, avgs, marker="o", linewidth=2, color="#1f77b4")
        plt.xlabel("IRCoT Generation Hop")
        plt.ylabel("Average Context Tokens")
        plt.title("Phase 2: Context Token Growth per Hop")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return  # only need one


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    payload: Dict,
    output_dir: Path,
    timestamp: str,
) -> None:
    """Persist JSON artifact and all Phase 2 plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"run_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logging.info("Saved run artifact: %s", json_path)

    metrics_map = payload.get("metrics_by_config", {})
    memory_map  = payload.get("memory_by_config", {})

    # Mask visualisation (SPIRE-Full)
    base_cfg = payload.get("base_config", {})
    mask_path = output_dir / f"mask_pattern_{timestamp}.png"
    SparseAttentionMask(
        sink_size=base_cfg.get("sink_size", 128),
        local_window=base_cfg.get("local_window", 2048),
        hash_budget=base_cfg.get("hash_budget", 256),
    ).visualize(seq_len=1000, save_path=mask_path)

    plot_f1_comparison(metrics_map, output_dir / f"f1_by_hop_{timestamp}.png")
    plot_context_tokens(metrics_map, output_dir / f"context_tokens_{timestamp}.png")
    plot_memory_comparison(memory_map, output_dir / f"memory_{timestamp}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 2: Dense IRCoT vs SPIRE baselines."""
    setup_logging()

    base_config = SPIREConfig(output_dir="results/phase2")
    output_dir  = base_config.output_path()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    evaluator = Evaluator()

    # ---- Load model (shared across all configs) ----
    logging.info("Loading model: %s", base_config.model_name)
    model_manager = ModelManager(base_config)

    # ---- Load dataset ----
    examples, gold_answers, hop_depths = load_musique_examples(base_config)
    logging.info("Running on %d examples", len(examples))

    # ---- Define the three configurations ----
    configs: List[Tuple[str, SPIREConfig]] = [
        (
            "dense",
            replace(base_config, use_sparse=False),
        ),
        (
            "spire_b5",
            replace(base_config, use_sparse=True, hash_budget=0),     # no hash — Sink+Local only
        ),
        (
            "spire_b6",
            replace(base_config, use_sparse=True, hash_budget=base_config.hash_budget),
        ),
    ]

    results_by_config:  Dict[str, List[Dict]] = {}
    metrics_by_config:  Dict[str, Dict]       = {}
    memory_by_config:   Dict[str, List[float]] = {}

    # ---- Try to skip dense re-run using Phase 1 results ----
    phase1_dir    = Path("results/phase1")
    phase1_loaded = try_load_phase1_dense(phase1_dir)

    for label, cfg in configs:
        if label == "dense" and phase1_loaded is not None:
            p1_results, p1_gold, p1_hops = phase1_loaded
            # Align to the current num_examples slice
            n = len(examples)
            results_by_config[label] = p1_results[:n]
            gold_slice = p1_gold[:n]
            hop_slice  = p1_hops[:n]
            metrics_by_config[label] = evaluator.evaluate_results(
                results=results_by_config[label],
                gold_answers=gold_slice,
                hop_depths=hop_slice,
            )
            memory_by_config[label] = []
            logging.info("[dense] Loaded %d examples from Phase 1 — skipping re-run.", n)
            continue

        results, avg_mem = run_one_config(cfg, model_manager, examples, label)
        results_by_config[label] = results
        memory_by_config[label]  = avg_mem

        gold_slice = gold_answers[: len(results)]
        hop_slice  = hop_depths[: len(results)]
        metrics_by_config[label] = evaluator.evaluate_results(
            results=results,
            gold_answers=gold_slice,
            hop_depths=hop_slice,
        )

        logging.info(
            "[%s] F1=%.4f  EM=%.4f",
            label,
            metrics_by_config[label]["overall_f1"],
            metrics_by_config[label]["overall_em"],
        )

    # ---- Save everything ----
    payload = {
        "base_config":       vars(base_config),
        "configs_run":       [label for label, _ in configs],
        "metrics_by_config": metrics_by_config,
        "memory_by_config":  memory_by_config,
        "results_by_config": results_by_config,
        "gold_answers":      gold_answers,
        "hop_depths":        hop_depths,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "model_name":        base_config.model_name,
        "phase":             2,
    }

    save_outputs(payload, output_dir, timestamp)

    # ---- Summary table ----
    logging.info("\n%s", "=" * 55)
    logging.info("%-15s  %8s  %8s", "Config", "F1", "EM")
    logging.info("-" * 55)
    for label in ["dense", "spire_b5", "spire_b6"]:
        if label in metrics_by_config:
            m = metrics_by_config[label]
            logging.info("%-15s  %8.4f  %8.4f", label, m["overall_f1"], m["overall_em"])
    logging.info("=" * 55)


if __name__ == "__main__":
    main()
