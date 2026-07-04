"""Phase 3: Full comparison — B1 / B2 / B3 / B6 / B7 on MuSiQue.

Baselines
─────────
B1  retrieve_once   Single BM25 retrieval + direct generation (lower bound).
B2  dense           IRCoT with full attention.  Loaded from Phase 1 if available.
B3  truncate_b3     IRCoT but accumulated context truncated to last 4 096 tokens.
B6  spire_b6        SPIRE-Full from Phase 2.  Loaded from Phase 2 if available.
B7  spire_attn      SPIRE-Full (sparse) + attention-guided retrieval (Phase 3).

Outputs (results/phase3/)
─────────────────────────
run_<timestamp>.json
f1_by_hop_<timestamp>.png          F1 vs hop depth for all 5 baselines
context_tokens_<timestamp>.png     Context growth (sanity check)
memory_<timestamp>.png             GPU memory per hop
attn_heatmap_<timestamp>.png       Attention heatmap for the first example (diagnostics)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import SPIREConfig
from src.attention_retriever import AttentionRetriever
from src.evaluator import Evaluator
from src.ircot_loop import IRCoTLoop
from src.model_manager import ModelManager
from src.retriever import BM25Retriever


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
# Dataset loading (reuse Phase 1 helpers)
# ---------------------------------------------------------------------------

def load_examples(config: SPIREConfig) -> Tuple[List[Dict], List[str], List[int]]:
    """Load, filter, and slice MuSiQue — identical pipeline to Phase 1."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import (
        filter_answerable,
        get_passages,
        infer_hop_depth,
        load_musique_validation,
    )

    examples = load_musique_validation()
    examples = filter_answerable(examples)
    examples = examples[: config.num_examples]

    gold_answers = [ex.get("answer", "") for ex in examples]
    hop_depths = [infer_hop_depth(ex) for ex in examples]
    return examples, gold_answers, hop_depths


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def gpu_memory_gb() -> float:
    """Return currently allocated GPU memory in GB (0 when CUDA unavailable)."""
    return torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0


# ---------------------------------------------------------------------------
# B1 — Retrieve-Once (no loop; single BM25 query + direct generation)
# ---------------------------------------------------------------------------

def run_retrieve_once(
    model_manager: ModelManager,
    examples: List[Dict],
    config: SPIREConfig,
) -> List[Dict]:
    """B1: Single retrieval on the question, then generate the answer directly."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import get_passages

    results: List[Dict] = []
    for example in tqdm(examples, desc="Phase 3 [B1 retrieve_once]"):
        question = example.get("question", "")
        passages = get_passages(example)
        if not question or not passages:
            continue

        retriever = BM25Retriever(passages=passages)
        retrieved = retriever.retrieve(question, top_k=config.retrieval_top_k)

        evidence = "\n\n".join(
            f"[Evidence {i + 1}] {p}" for i, p in enumerate(retrieved)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question based on the evidence. "
                    "Write 'So the answer is: <answer>'."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:",
            },
        ]
        response = model_manager.generate(messages, max_new_tokens=config.max_new_tokens)

        import re
        match = re.search(
            r"(?:the answer is|answer is)[:\s]*(.+?)(?:\.|$)",
            response,
            flags=re.IGNORECASE,
        )
        answer = match.group(1).strip() if match else response.strip()
        ctx_len = model_manager.get_context_length_from_messages(messages)

        results.append(
            {
                "question": question,
                "answer": answer,
                "num_hops": 1,
                "reasoning_chain": [response],
                "retrieved_passages": retrieved,
                "context_tokens_per_hop": [ctx_len],
                "total_tokens": ctx_len,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Generic IRCoT runner (B2 / B3 / B7)
# ---------------------------------------------------------------------------

def run_ircot(
    config: SPIREConfig,
    model_manager: ModelManager,
    examples: List[Dict],
    label: str,
    use_attention: bool = False,
) -> Tuple[List[Dict], List[float]]:
    """Run IRCoT on all examples under the given config.

    Args:
        config:         SPIREConfig instance for this run (all flags set).
        model_manager:  Shared model.
        examples:       MuSiQue examples.
        label:          Display label for tqdm.
        use_attention:  When True, builds AttentionRetriever per example.

    Returns:
        (results, avg_gpu_memory_per_hop)
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import get_passages

    results: List[Dict] = []
    memory_by_hop: Dict[int, List[float]] = defaultdict(list)

    for example in tqdm(examples, desc=f"Phase 3 [{label}]"):
        question = example.get("question", "")
        passages = get_passages(example)
        if not question or not passages:
            continue

        bm25 = BM25Retriever(passages=passages)

        attn_retriever: Optional[AttentionRetriever] = None
        if use_attention:
            attn_retriever = AttentionRetriever(
                model=model_manager,
                passages=passages,
                bm25_retriever=bm25,
                max_seq_len=config.attn_retrieval_max_seq,
            )

        loop = IRCoTLoop(
            model=model_manager,
            retriever=bm25,
            config=config,
            attention_retriever=attn_retriever,
        )

        result = _run_with_memory(loop, question, memory_by_hop)
        results.append(result)

    avg_mem = [
        float(sum(memory_by_hop[h]) / len(memory_by_hop[h]))
        for h in sorted(memory_by_hop.keys())
    ]
    return results, avg_mem


def _run_with_memory(
    loop: IRCoTLoop,
    question: str,
    memory_by_hop: Dict[int, List[float]],
) -> Dict:
    """Run the IRCoT loop and record GPU memory at each hop."""
    from src.sparse_attention import SparseAttentionMask

    config = loop.config
    retrieved_so_far: List[str] = []
    reasoning_so_far: List[str] = []
    context_tokens_per_hop: List[int] = []
    current_query = question
    answer = ""

    for hop_idx in range(config.max_hops):
        if config.use_attention_retrieval and loop.attention_retriever is not None:
            passages = loop.attention_retriever.retrieve(
                question=question,
                reasoning_so_far=reasoning_so_far,
                top_k=config.retrieval_top_k,
            )
        else:
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
                messages=messages, max_new_tokens=config.max_new_tokens
            )

        memory_by_hop[hop_idx + 1].append(max(mem_before, gpu_memory_gb()))
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
# Load prior phase results (skip re-runs)
# ---------------------------------------------------------------------------

def try_load_prior(
    results_dir: Path, n: int
) -> Optional[Tuple[List[Dict], List[str], List[int]]]:
    """Load the most recent run JSON from results_dir; return first n entries."""
    candidates = sorted(results_dir.glob("run_*.json"), reverse=True)
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    results = payload.get("results", [])
    golds = payload.get("gold_answers", [])
    hops = payload.get("hop_depths", [])
    if results and golds:
        return results[:n], golds[:n], hops[:n]

    # Phase 2 format stores results per config
    rbc = payload.get("results_by_config", {})
    for key in ("dense", "spire_b6"):
        if key in rbc and rbc[key]:
            return rbc[key][:n], payload.get("gold_answers", [])[:n], payload.get("hop_depths", [])[:n]
    return None


# ---------------------------------------------------------------------------
# Attention heatmap (diagnostic — first example only)
# ---------------------------------------------------------------------------

def save_attention_heatmap(
    model_manager: ModelManager,
    example: Dict,
    output_path: Path,
) -> None:
    """Run a forward pass on the first example and save an attention heatmap.

    Uses the last target layer, averaged over heads.  This is a diagnostic
    figure showing which passage tokens the reasoning tokens attended to most.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_phase1 import get_passages

    question = example.get("question", "")
    passages = get_passages(example)
    if not question or not passages:
        return

    bm25 = BM25Retriever(passages=passages)
    # Use a short dummy reasoning for the heatmap
    reasoning_text = f"To answer '{question}', I need to find relevant evidence."

    retriever = AttentionRetriever(
        model=model_manager, passages=passages[:5], bm25_retriever=bm25
    )
    input_ids, passage_ranges, reasoning_range = retriever._build_input(
        question=question, reasoning_text=reasoning_text
    )

    if input_ids.shape[-1] > AttentionRetriever.__init__.__defaults__[0] if False else 1500:
        logging.info("Skipping heatmap — sequence too long.")
        return

    device = model_manager.model.device
    with torch.no_grad():
        outputs = model_manager.model(
            input_ids=input_ids.to(device), output_attentions=True
        )

    # Take the last target layer
    last_layer_attn = outputs.attentions[-1][0].cpu().float()  # (heads, seq, seq)
    avg_attn = last_layer_attn.mean(dim=0).numpy()  # (seq, seq)

    r_start, r_end = reasoning_range
    seq_len = avg_attn.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_attn[r_start:r_end, :], aspect="auto", cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xlabel("Key token position")
    ax.set_ylabel("Reasoning token position")
    ax.set_title(
        f"Attention heatmap — reasoning tokens attending to context\n"
        f"seq_len={seq_len}  reasoning_tokens={r_end - r_start}"
    )

    # Mark passage boundaries
    for p_start, p_end in passage_ranges:
        ax.axvline(x=p_start, color="cyan", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    del outputs
    logging.info("Saved attention heatmap: %s", output_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_f1_comparison(metrics_map: Dict[str, Dict], output_path: Path) -> None:
    """Plot F1 vs hop depth for all 5 baselines."""
    styles = {
        "retrieve_once": ("D", "#9467bd", ":", "B1 Retrieve-Once"),
        "dense":         ("o", "#1f77b4", "-", "B2 Dense IRCoT"),
        "truncate_b3":   ("v", "#d62728", "--", "B3 IRCoT-Truncate"),
        "spire_b6":      ("^", "#2ca02c", "-.", "B6 SPIRE-Full"),
        "spire_attn":    ("s", "#ff7f0e", "-", "B7 SPIRE+Attention"),
    }

    plt.figure(figsize=(10, 5))
    for key, metrics in metrics_map.items():
        f1_by_hops = metrics.get("f1_by_hops", {})
        if not f1_by_hops:
            continue
        hops = sorted(int(k) for k in f1_by_hops)
        values = [float(f1_by_hops[h]) for h in hops]
        marker, color, ls, label = styles.get(key, ("o", "gray", "-", key))
        plt.plot(hops, values, marker=marker, linestyle=ls, color=color, linewidth=2, label=label)

    plt.xlabel("Hop Depth")
    plt.ylabel("F1")
    plt.title("Phase 3: F1 vs Hop Depth — All Baselines")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory(memory_map: Dict[str, List[float]], output_path: Path) -> None:
    """Plot GPU memory per hop for all configurations."""
    if not any(memory_map.values()):
        return
    styles = {
        "dense":       ("#1f77b4", "-",  "B2 Dense"),
        "truncate_b3": ("#d62728", "--", "B3 Truncate"),
        "spire_b6":    ("#2ca02c", "-.", "B6 SPIRE-Full"),
        "spire_attn":  ("#ff7f0e", "-",  "B7 SPIRE+Attn"),
    }
    plt.figure(figsize=(10, 5))
    for key, vals in memory_map.items():
        if not vals:
            continue
        color, ls, label = styles.get(key, ("gray", "-", key))
        plt.plot(range(1, len(vals) + 1), vals, linestyle=ls, color=color, linewidth=2, label=label)
    plt.xlabel("Hop")
    plt.ylabel("GPU Memory (GB)")
    plt.title("Phase 3: GPU Memory per Hop")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_context_tokens(metrics_map: Dict[str, Dict], output_path: Path) -> None:
    """Plot average context growth (any config — for sanity check)."""
    for metrics in metrics_map.values():
        tokens_by_hop = metrics.get("context_tokens_by_hop", {})
        if not tokens_by_hop:
            continue
        hops = sorted(int(k) for k in tokens_by_hop)
        avgs = [sum(tokens_by_hop[h]) / len(tokens_by_hop[h]) for h in hops]
        plt.figure(figsize=(10, 5))
        plt.plot(hops, avgs, marker="o", linewidth=2, color="#1f77b4")
        plt.xlabel("IRCoT Generation Hop")
        plt.ylabel("Average Context Tokens")
        plt.title("Phase 3: Context Growth per Hop")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return


# ---------------------------------------------------------------------------
# Save all outputs
# ---------------------------------------------------------------------------

def save_outputs(payload: Dict, output_dir: Path, timestamp: str) -> None:
    """Persist JSON run artifact and all Phase 3 plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"run_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logging.info("Saved run artifact: %s", json_path)

    metrics_map = payload.get("metrics_by_config", {})
    memory_map = payload.get("memory_by_config", {})

    plot_f1_comparison(metrics_map, output_dir / f"f1_by_hop_{timestamp}.png")
    plot_context_tokens(metrics_map, output_dir / f"context_tokens_{timestamp}.png")
    plot_memory(memory_map, output_dir / f"memory_{timestamp}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 3: five-way comparison including attention-guided retrieval."""
    setup_logging()

    base_config = SPIREConfig(output_dir="results/phase3")
    output_dir = base_config.output_path()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = base_config.num_examples

    evaluator = Evaluator()

    # ---- Load model ----
    logging.info("Loading model: %s", base_config.model_name)
    model_manager = ModelManager(base_config)

    # ---- Load dataset ----
    examples, gold_answers, hop_depths = load_examples(base_config)
    logging.info("Running on %d examples", len(examples))

    results_by_config: Dict[str, List[Dict]] = {}
    metrics_by_config: Dict[str, Dict] = {}
    memory_by_config: Dict[str, List[float]] = {}

    # ----
    # B1 — Retrieve-Once
    # ----
    b1_results = run_retrieve_once(model_manager, examples, base_config)
    results_by_config["retrieve_once"] = b1_results
    metrics_by_config["retrieve_once"] = evaluator.evaluate_results(
        b1_results, gold_answers[:len(b1_results)], hop_depths[:len(b1_results)]
    )
    memory_by_config["retrieve_once"] = []
    logging.info("[B1] F1=%.4f  EM=%.4f", metrics_by_config["retrieve_once"]["overall_f1"],
                 metrics_by_config["retrieve_once"]["overall_em"])

    # ----
    # B2 — Dense IRCoT (load Phase 1 if available)
    # ----
    p1_loaded = try_load_prior(Path("results/phase1"), n)
    if p1_loaded is not None:
        r, g, h = p1_loaded
        results_by_config["dense"] = r
        metrics_by_config["dense"] = evaluator.evaluate_results(r, g, h)
        memory_by_config["dense"] = []
        logging.info("[B2 dense] Loaded from Phase 1 (%d examples).", len(r))
    else:
        cfg_b2 = replace(base_config, use_sparse=False, use_attention_retrieval=False,
                         truncate_context_tokens=0)
        r, mem = run_ircot(cfg_b2, model_manager, examples, "B2 dense")
        results_by_config["dense"] = r
        memory_by_config["dense"] = mem
        metrics_by_config["dense"] = evaluator.evaluate_results(
            r, gold_answers[:len(r)], hop_depths[:len(r)]
        )
        logging.info("[B2 dense] F1=%.4f", metrics_by_config["dense"]["overall_f1"])

    # ----
    # B3 — IRCoT-Truncate (last 4 096 tokens)
    # ----
    cfg_b3 = replace(base_config, use_sparse=False, use_attention_retrieval=False,
                     truncate_context_tokens=4096)
    r3, mem3 = run_ircot(cfg_b3, model_manager, examples, "B3 truncate")
    results_by_config["truncate_b3"] = r3
    memory_by_config["truncate_b3"] = mem3
    metrics_by_config["truncate_b3"] = evaluator.evaluate_results(
        r3, gold_answers[:len(r3)], hop_depths[:len(r3)]
    )
    logging.info("[B3] F1=%.4f  EM=%.4f", metrics_by_config["truncate_b3"]["overall_f1"],
                 metrics_by_config["truncate_b3"]["overall_em"])

    # ----
    # B6 — SPIRE-Full (load Phase 2 if available)
    # ----
    p2_loaded = try_load_prior(Path("results/phase2"), n)
    if p2_loaded is not None:
        r, g, h = p2_loaded
        results_by_config["spire_b6"] = r
        metrics_by_config["spire_b6"] = evaluator.evaluate_results(r, g, h)
        memory_by_config["spire_b6"] = []
        logging.info("[B6 spire_b6] Loaded from Phase 2 (%d examples).", len(r))
    else:
        cfg_b6 = replace(base_config, use_sparse=True, hash_budget=256,
                         use_attention_retrieval=False, truncate_context_tokens=0)
        r6, mem6 = run_ircot(cfg_b6, model_manager, examples, "B6 spire_b6")
        results_by_config["spire_b6"] = r6
        memory_by_config["spire_b6"] = mem6
        metrics_by_config["spire_b6"] = evaluator.evaluate_results(
            r6, gold_answers[:len(r6)], hop_depths[:len(r6)]
        )
        logging.info("[B6] F1=%.4f", metrics_by_config["spire_b6"]["overall_f1"])

    # ----
    # B7 — SPIRE + Attention Retrieval (Phase 3 novel contribution)
    # ----
    cfg_b7 = replace(base_config, use_sparse=True, hash_budget=256,
                     use_attention_retrieval=True, truncate_context_tokens=0)
    r7, mem7 = run_ircot(cfg_b7, model_manager, examples, "B7 spire_attn",
                          use_attention=True)
    results_by_config["spire_attn"] = r7
    memory_by_config["spire_attn"] = mem7
    metrics_by_config["spire_attn"] = evaluator.evaluate_results(
        r7, gold_answers[:len(r7)], hop_depths[:len(r7)]
    )
    logging.info("[B7] F1=%.4f  EM=%.4f", metrics_by_config["spire_attn"]["overall_f1"],
                 metrics_by_config["spire_attn"]["overall_em"])

    # ---- Attention heatmap (first example, diagnostic) ----
    heatmap_path = output_dir / f"attn_heatmap_{timestamp}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    if examples:
        try:
            save_attention_heatmap(model_manager, examples[0], heatmap_path)
        except Exception as exc:
            logging.warning("Heatmap generation skipped: %s", exc)

    # ---- Save results ----
    payload = {
        "base_config":       vars(base_config),
        "configs_run":       list(results_by_config.keys()),
        "metrics_by_config": metrics_by_config,
        "memory_by_config":  memory_by_config,
        "results_by_config": results_by_config,
        "gold_answers":      gold_answers,
        "hop_depths":        hop_depths,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "model_name":        base_config.model_name,
        "phase":             3,
    }
    save_outputs(payload, output_dir, timestamp)

    # ---- Summary table ----
    logging.info("\n%s", "=" * 60)
    logging.info("%-18s  %8s  %8s", "Config", "F1", "EM")
    logging.info("-" * 60)
    for key in ["retrieve_once", "dense", "truncate_b3", "spire_b6", "spire_attn"]:
        if key in metrics_by_config:
            m = metrics_by_config[key]
            logging.info("%-18s  %8.4f  %8.4f", key, m["overall_f1"], m["overall_em"])
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
