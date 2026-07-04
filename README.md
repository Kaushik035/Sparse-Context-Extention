# SPIRE — Scalable Interleaved Retrieval-Reasoning via Sparse Attention

> **Undergraduate Summer Research Project · B.Tech 3rd Year · IIT Bombay · Summer 2026**  
> Topic: Optimizing Context Window Usage in LLMs for Multi-Hop RAG

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Phase 1 — IRCoT Baseline and Profiling](#phase-1--ircot-baseline-and-profiling)
  - [What Phase 1 Does](#what-phase-1-does)
  - [Architecture and Data Flow](#architecture-and-data-flow)
  - [Module Breakdown](#module-breakdown)
  - [Outputs](#outputs)
- [Phase 2 — Sparse Attention Integration](#phase-2--sparse-attention-integration)
  - [Why Phase 2 Is Needed](#why-phase-2-is-needed)
  - [The SPIRE Sparse Attention Policy](#the-spire-sparse-attention-policy)
  - [How Sparse Generation Works Technically](#how-sparse-generation-works-technically)
  - [Files Added or Patched in Phase 2](#files-added-or-patched-in-phase-2)
  - [Baselines Run in Phase 2](#baselines-run-in-phase-2)
  - [Phase 2 Data Flow](#phase-2-data-flow)
  - [Phase 2 Outputs](#phase-2-outputs)
- [Setup and Installation](#setup-and-installation)
- [Running Phase 1](#running-phase-1)
- [Running Phase 2](#running-phase-2)
- [Running on a GPU / Cloud Machine](#running-on-a-gpu--cloud-machine)
- [Configuration Reference](#configuration-reference)
- [Security Notes](#security-notes)

---

## Project Overview

**SPIRE** addresses a core bottleneck in multi-hop Retrieval-Augmented Generation (RAG): every time the model retrieves a new passage and reasons over it, the context window grows. After 4–5 hops, the accumulated context (passages + reasoning steps) can reach 10K–20K tokens, and the model gives every token identical dense attention — including stale evidence from earlier hops that is no longer critical.

SPIRE's solution is a structured **sparse attention policy**:

| Context region | Attention type | Rationale |
|---|---|---|
| System prompt + question | Always attended (sink) | Anchor for all reasoning |
| Current hop's passage | Dense (local window) | Fresh evidence — highest priority |
| All prior accumulated hops | Sparse (hash-selected) | Still accessible, but low budget cost |

This allows more productive reasoning hops to fit into the same context window without the accuracy degradation caused by attention dilution.

**The project is implemented in three phases:**

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Reproduce IRCoT on MuSiQue, profile context growth and F1 degradation | **Complete** |
| Phase 2 | Add sparse attention (SPIRE) and compare against dense baselines | **Complete** |
| Phase 3 | Replace BM25 with attention-guided retrieval | Planned |

---

## Repository Structure

```
SURP/
├── SPIRE.md                        # Full research proposal
├── SPIRE_Implementation_Prompt.md  # Engineering specification
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.py                       # Shared dataclass configuration
├── .env                            # HuggingFace token (never commit)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── model_manager.py            # Model loading, dense generation, sparse generation
│   ├── retriever.py                # BM25 retriever
│   ├── ircot_loop.py               # IRCoT loop — switches between dense/sparse via config
│   ├── evaluator.py                # F1 / EM evaluation utilities
│   └── sparse_attention.py         # Phase 2: SPIRE sparse mask builder + visualiser
│
├── scripts/
│   ├── run_phase1.py               # Phase 1: dense IRCoT baseline
│   ├── run_phase2.py               # Phase 2: dense vs SPIRE-B5 vs SPIRE-B6
│   └── run_phase3.py               # Phase 3 (planned)
│
├── data/
│   └── musique/                    # Auto-downloaded on first run
│
└── results/
    ├── phase1/                     # Phase 1 JSON artifacts + plots
    └── phase2/                     # Phase 2 JSON artifacts + comparison plots
```

---

## Phase 1 — IRCoT Baseline and Profiling

### What Phase 1 Does

Phase 1 establishes the **dense IRCoT baseline** — the primary system SPIRE is compared against. It:

1. Loads the LLM and tokenizer from HuggingFace.
2. Downloads the MuSiQue multi-hop QA dataset (automatically on first run).
3. For each question, builds a BM25 index over that question's passage pool.
4. Runs the full IRCoT loop: retrieve → reason → retrieve → reason → ... up to `max_hops`.
5. Extracts the final answer and measures token-level F1 against the gold answer.
6. Profiles **context token count at every hop** — the key diagnostic for showing where the window fills up.
7. Saves a JSON run artifact and generates two plots.

The goal is to empirically confirm that: (a) F1 degrades at deeper hops, and (b) context size grows with each hop — establishing the problem that SPIRE targets.

---

### Architecture and Data Flow

```
Question (from MuSiQue)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  IRCoT Loop  (src/ircot_loop.py)                                 │
│                                                                  │
│  ┌──── Hop k ────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │  current_query ──► BM25Retriever.retrieve()               │  │
│  │                         │                                 │  │
│  │                    top-k passages                         │  │
│  │                         │                                 │  │
│  │                    accumulated into retrieved_so_far      │  │
│  │                         │                                 │  │
│  │  Build prompt:                                            │  │
│  │    [SYSTEM] Answer step by step...                        │  │
│  │    [USER]   Question: ...                                 │  │
│  │             Evidence: [Evidence 1] ... [Evidence k*3] ... │  │
│  │             Previous reasoning: Step 1... Step k-1...     │  │
│  │             Continue reasoning step by step.              │  │
│  │                         │                                 │  │
│  │  ModelManager.generate() ──► LLM response rₖ             │  │
│  │                         │                                 │  │
│  │  Track context_tokens_per_hop[k]                          │  │
│  │                         │                                 │  │
│  │  If "the answer is" in rₖ ──► extract answer, stop        │  │
│  │  Else: current_query = rₖ   ──► next hop                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
Evaluator.f1_score(predicted, gold)
        │
        ▼
results/phase1/run_<timestamp>.json
results/phase1/f1_by_hop_<timestamp>.png
results/phase1/context_tokens_by_hop_<timestamp>.png
```

---

### Module Breakdown

#### `config.py` — Shared Configuration

A Python `dataclass` that holds all tunable parameters. A single instance is created at the top of every run script and passed to all modules. No magic globals.

```python
@dataclass
class SPIREConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  # swap to 8B on GPU
    max_hops: int = 4
    max_new_tokens: int = 256
    retrieval_top_k: int = 3
    num_examples: int = 10   # set to 100 for real experiments
    output_dir: str = "results/phase1"
    # Phase 2 sparse attention hyperparameters (unused in Phase 1)
    sink_size: int = 128
    local_window: int = 2048
    hash_budget: int = 256
    use_sparse: bool = False
```

---

#### `src/model_manager.py` — Model Lifecycle

Loads the HuggingFace model and tokenizer once. Exposes three methods used by the rest of the system:

| Method | Purpose |
|---|---|
| `generate(messages)` | Run one forward + decode pass; return generated text only |
| `count_tokens(text)` | Count tokens in a plain string |
| `get_context_length_from_messages(messages)` | Token count for a full chat-formatted prompt |

Key implementation details:
- Token from `.env` is loaded via `os.environ["HF_TOKEN"]` — never hardcoded.
- `device_map="auto"` places the model on whatever hardware is available (GPU, CPU, or split across both).
- `apply_chat_template` correctly formats messages with Llama's `<|begin_of_text|>` / `<|eot_id|>` tokens.
- Handles both tensor and `BatchEncoding` return types from newer `transformers` versions.

---

#### `src/retriever.py` — BM25 Retriever

A thin wrapper around `rank_bm25.BM25Okapi`. For each MuSiQue example, a **fresh retriever is built** from that example's passage pool (typically 20 paragraphs including distractors). This matches the IRCoT setup from the original paper.

```python
retriever = BM25Retriever(passages=example_passages)
top_k = retriever.retrieve(query=current_reasoning_step, top_k=3)
```

Tokenization is whitespace-split lowercase — intentionally simple, matching standard BM25 usage.

---

#### `src/ircot_loop.py` — IRCoT Loop

The core of Phase 1. At each hop:

1. BM25 retrieves `top_k` passages using the current query (initially the question itself, then each reasoning step).
2. All retrieved passages are **accumulated** into the prompt — this is what causes context growth.
3. The full prompt (question + all evidence + all prior reasoning) is sent to the LLM.
4. If the response contains `"the answer is"`, the answer is extracted and the loop stops.
5. Otherwise, the response becomes the next retrieval query.

The loop tracks `context_tokens_per_hop` — the token count of the full prompt at each hop — for profiling.

---

#### `src/evaluator.py` — Evaluation

Implements **SQuAD-style** token-level F1 and Exact Match:

- **Normalize**: lowercase, strip articles (`a/an/the`), strip punctuation, collapse whitespace.
- **F1**: token overlap between predicted and gold answer (precision × recall harmonic mean).
- **EM**: exact match after normalization.

Additionally computes:
- `f1_by_hops` — average F1 grouped by the gold hop-depth of each question (2-hop, 3-hop, 4-hop).
- `context_tokens_by_hop` — list of token counts at each generation hop across all examples.

---

#### `scripts/run_phase1.py` — Experiment Entrypoint

Ties everything together:

1. Loads config, initializes model.
2. Loads MuSiQue validation split (tries HuggingFace Hub first; falls back to Google Drive download automatically).
3. Filters to answerable examples only.
4. For each example: builds a fresh BM25 retriever, runs the IRCoT loop, collects results.
5. Evaluates with `Evaluator`.
6. Saves `results/phase1/run_<timestamp>.json` with full config, per-example results, and aggregated metrics.
7. Saves two plots to `results/phase1/`.

---

### Outputs

After a successful Phase 1 run, the following files appear under `results/phase1/`:

| File | Contents |
|---|---|
| `run_<timestamp>.json` | Full run artifact: config, per-example results (question, predicted answer, reasoning chain, retrieved passages, context tokens per hop), aggregate metrics |
| `f1_by_hop_<timestamp>.png` | F1 score at each hop depth — shows where accuracy degrades |
| `context_tokens_by_hop_<timestamp>.png` | Average context token count at each IRCoT generation hop — shows window growth |

The JSON structure:
```json
{
  "config": { ... },
  "metrics": {
    "overall_f1": 0.0535,
    "overall_em": 0.0,
    "f1_by_hops": { "2": 0.07, "3": 0.04, "4": 0.03 },
    "context_tokens_by_hop": { "1": [312, 298, ...], "2": [601, 589, ...] }
  },
  "results": [ { "question": "...", "answer": "...", "num_hops": 3, ... } ],
  "timestamp": "2026-07-04T14:54:21",
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "phase": 1
}
```

---

## Phase 2 — Sparse Attention Integration

### Why Phase 2 Is Needed

Phase 1 established the problem: as the IRCoT loop accumulates more retrieved passages and reasoning steps, the context window fills up, and the model gives **equal dense attention to every single token** — including stale evidence from hop 1 when it is now working on hop 4. This wastes the limited attention budget on content that is no longer critical and dilutes focus on the fresh evidence that matters most right now.

Phase 2 addresses this by applying a **structured sparse attention policy** during generation, allowing the model to focus its attention budget on what matters while still being able to access all past context if truly needed. Unlike truncation (which throws away old tokens entirely), sparse attention keeps all tokens in the window but dramatically reduces how much "budget" they consume.

---

### The SPIRE Sparse Attention Policy

At every generation step inside the IRCoT loop, the SPIRE policy divides the accumulated context into three regions and assigns a different attention behaviour to each:

```
Context window at hop k:

┌─────────────────────────────────────────────────────────────────┐
│  [SYS + Q]  │  p₁ r₁  │  p₂ r₂  │  ...  │  pₖ₋₁ rₖ₋₁  │  pₖ  │
└─────────────────────────────────────────────────────────────────┘
  ↑ SINK        ↑──── MIDDLE (old cycles) ────↑   ↑ LOCAL WINDOW ↑
  always dense   sparse (random hash budget)       always dense
```

| Region | Policy | Size | Why |
|---|---|---|---|
| **Sink** | Always attended | First `sink_size=128` tokens | System prompt + question are the anchor for every reasoning step — they must never be masked out |
| **Local window** | Always attended (dense) | Last `local_window=2048` tokens | The current hop's retrieved passage + the reasoning being generated right now — highest priority, always needs full focus |
| **Middle** | Sparse (random / hash selection) | `hash_budget=256` tokens selected from all older cycles | Old passages and reasoning steps are still accessible for reference but don't consume the full attention budget |

**Why not just truncate?** Truncation is irreversible — if an earlier hop contained a critical entity, truncating it means the model permanently loses access to it. Sparse attention keeps every token in the window (no information loss) but assigns them a lower attention budget.

**Why not summarise?** Summarisation introduces its own compression loss (and is itself a baseline — B4 in the experiments). It also adds generation cost at every hop.

---

### How Sparse Generation Works Technically

The HuggingFace `generate()` API does not accept a 2D causal attention mask. Phase 2 uses a **manual KV-cached generation loop** instead:

```
Step 1 — Dense Prefill:
  model.forward(full_prompt, use_cache=True)
  → Builds the KV cache for all prompt tokens with FULL dense attention.
  → Returns the KV cache + logits for the first generated token.
  → Why dense here? The prompt just needs to be encoded; the attention
    pattern that matters for research is what happens during generation.

Step 2 — Sparse Generation Loop (per token):
  for each new token at position t:
    sparse_mask = build_generation_mask(total_len=input_len + t + 1)
    # sparse_mask is a 1D tensor: 1 = attend, 0 = skip
    # Covers: sink positions + local window + randomly-selected middle positions

    model.forward(
        input_ids  = next_token,          # just 1 new token
        attention_mask = sparse_mask,     # (1, total_len) — controls KV access
        past_key_values = kv_cache,       # all past KV states still available
        use_cache = True,
    )
```

Key properties of this design:
- **KV cache is used** — each step processes only 1 new token, not the full sequence. This keeps generation speed reasonable.
- **All past KV states remain in memory** — the sparse mask controls *which* KV states to attend to, but the states themselves are not deleted. No information loss.
- **Sink + local positions always get mask=1** — they receive full attention at every generation step.
- **Middle positions get mask=1 for only `hash_budget` randomly-chosen positions** — the rest get mask=0 and are skipped.
- **Backward compatible** — `use_sparse=False` in `config.py` routes the loop through the original `model.generate()` call, giving identical Phase 1 behaviour.

---

### Files Added or Patched in Phase 2

#### `src/sparse_attention.py` — NEW

The `SparseAttentionMask` class. The core of Phase 2.

| Method | Purpose |
|---|---|
| `build_mask(seq_len)` | Build a full 2D boolean causal mask — used for visualisation and analysis |
| `build_generation_mask(total_len)` | Build the 1D mask for a single KV-cached generation step. Returns a `LongTensor` of shape `(total_len,)`: 1 = attend, 0 = skip |
| `sparsity(seq_len)` | Return the fraction of causal attention entries that are skipped at a given sequence length |
| `visualize(seq_len, save_path)` | Save a heatmap of the attention pattern to a PNG file |

The three mask components in `build_generation_mask`:
```python
# Sink: first sink_size tokens always 1
mask[:min(self.sink_size, total_len)] = 1

# Local window: last local_window tokens always 1
local_start = max(0, new_pos - self.local_window + 1)
mask[local_start:total_len] = 1

# Hash / random: select hash_budget positions from the middle gap
middle_positions = torch.arange(middle_start, middle_end)
perm = torch.randperm(len(middle_positions))[:hash_budget]
mask[middle_positions[perm]] = 1
```

#### `src/model_manager.py` — PATCHED

Added `generate_with_sparse_mask(messages, mask_builder, max_new_tokens)`. No existing methods changed — Phase 1 behaviour is fully preserved.

#### `src/ircot_loop.py` — PATCHED

A 3-line branch was inserted before the `model.generate()` call:
```python
if self.config.use_sparse:
    # Phase 2: sparse generation
    mask_builder = SparseAttentionMask(sink_size, local_window, hash_budget)
    response = self.model.generate_with_sparse_mask(messages, mask_builder, max_new_tokens)
else:
    # Phase 1: dense generation — identical to original
    response = self.model.generate(messages, max_new_tokens)
```

#### `config.py` — PATCHED

Added `use_attention_retrieval: bool = False` (Phase 3 placeholder). The existing sparse fields (`sink_size`, `local_window`, `hash_budget`, `use_sparse`) were already present as Phase 1 placeholders and are now active.

#### `scripts/run_phase2.py` — NEW

Orchestrates the three-way comparison. Key logic:
1. Tries to load the most recent Phase 1 JSON from `results/phase1/` as the dense baseline — avoids re-running B2 if results already exist.
2. Runs B5 (`use_sparse=True, hash_budget=0`) — Sink + Local only, no middle-region selection.
3. Runs B6 (`use_sparse=True, hash_budget=256`) — full SPIRE with random hash selection.
4. Tracks GPU memory at each hop via `torch.cuda.memory_allocated()`.
5. Saves a single JSON artifact + 4 plots.

---

### Baselines Run in Phase 2

| ID | Name | Config | Description |
|---|---|---|---|
| B2 | Dense IRCoT | `use_sparse=False` | Full dense attention — loaded from Phase 1 results if available |
| B5 | SPIRE Sink+Local | `use_sparse=True, hash_budget=0` | Ablation: only sink + local window, no sparse selection from old cycles |
| B6 | SPIRE Full | `use_sparse=True, hash_budget=256` | Full SPIRE: sink + local + random selection from middle |

B5 is an ablation that answers the question: *does the hash/random selection from old cycles actually contribute, or does the local window alone explain any gains?*

---

### Phase 2 Data Flow

```
Same question/dataset loading as Phase 1
        │
        ├──► [B2 Dense]    Load from results/phase1/ (skip re-run)
        │
        ├──► [B5 Sink+Local]
        │        │
        │    IRCoT loop  (use_sparse=True, hash_budget=0)
        │        │
        │    generate_with_sparse_mask()
        │        │
        │    ┌── Prefill: dense forward, build KV cache
        │    └── Per-token loop:
        │            mask = sink(128) ∪ local(2048)  [no middle]
        │            model.forward(token, mask, past_kv)
        │
        └──► [B6 SPIRE Full]
                 │
             IRCoT loop  (use_sparse=True, hash_budget=256)
                 │
             generate_with_sparse_mask()
                 │
             ┌── Prefill: dense forward, build KV cache
             └── Per-token loop:
                     mask = sink(128) ∪ local(2048) ∪ random(256 from middle)
                     model.forward(token, mask, past_kv)
        │
        ▼
Evaluator  →  f1_by_hops, overall_f1, overall_em
        │
        ▼
results/phase2/run_<timestamp>.json
results/phase2/f1_by_hop_<timestamp>.png        (B2 vs B5 vs B6 on same axes)
results/phase2/context_tokens_<timestamp>.png   (context growth, sanity check)
results/phase2/memory_<timestamp>.png           (GPU memory per hop)
results/phase2/mask_pattern_<timestamp>.png     (SPIRE attention pattern heatmap)
```

---

### Phase 2 Outputs

| File | Contents |
|---|---|
| `run_<timestamp>.json` | Per-example results for all 3 configs + aggregated metrics + GPU memory per hop |
| `f1_by_hop_<timestamp>.png` | **Key comparison figure** — F1 at each hop depth for B2 / B5 / B6 on one plot |
| `context_tokens_<timestamp>.png` | Average context token count per hop (same across configs — confirms window growth) |
| `memory_<timestamp>.png` | GPU memory allocated (GB) at each hop — shows memory cost difference |
| `mask_pattern_<timestamp>.png` | Heatmap of the SPIRE sparse attention pattern for seq_len=1000 — shows sparsity visually |

The JSON structure for Phase 2:
```json
{
  "base_config": { "sink_size": 128, "local_window": 2048, "hash_budget": 256, ... },
  "configs_run": ["dense", "spire_b5", "spire_b6"],
  "metrics_by_config": {
    "dense":    { "overall_f1": 0.xx, "f1_by_hops": { "2": ..., "3": ..., "4": ... } },
    "spire_b5": { "overall_f1": 0.xx, "f1_by_hops": { ... } },
    "spire_b6": { "overall_f1": 0.xx, "f1_by_hops": { ... } }
  },
  "memory_by_config": {
    "dense":    [GB_hop1, GB_hop2, ...],
    "spire_b5": [...],
    "spire_b6": [...]
  },
  "results_by_config": { "dense": [...], "spire_b5": [...], "spire_b6": [...] },
  "phase": 2
}
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- A HuggingFace account with a **Read** token
- Llama-3.1-8B-Instruct license accepted at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### Step 1 — Clone / open the project

```cmd
cd C:\Users\<you>\Desktop\SURP
```

### Step 2 — Create and activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 4 — Add your HuggingFace token

Create (or edit) `.env` in the project root:

```
HF_TOKEN=hf_your_token_here
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`.

---

## Running Phase 1

From the project root (with the virtual environment active):

```cmd
python scripts/run_phase1.py
```

**What happens on first run:**
1. Downloads `Llama-3.2-1B-Instruct` weights (~2.5 GB) from HuggingFace into `~/.cache/huggingface/`.
2. Attempts to load MuSiQue from HuggingFace Hub; falls back to downloading `musique_v1.0.zip` (~272 MB) from Google Drive into `data/musique/` automatically.
3. Runs the IRCoT loop on `num_examples` questions.
4. Saves results and plots to `results/phase1/`.

**On subsequent runs**, cached model weights and the dataset are reused — startup takes only a few seconds.

---

## Running Phase 2

From the project root (with the virtual environment active):

```cmd
python scripts/run_phase2.py
```

**What the script does automatically:**
1. Loads the model (shared across all three configs — loaded once).
2. Loads MuSiQue (uses the cached download from Phase 1).
3. Checks `results/phase1/` for an existing run artifact — if found, reuses those dense B2 results instead of re-running, saving ~half the total time.
4. Runs B5 (Sink+Local sparse) on all `num_examples` questions.
5. Runs B6 (Full SPIRE sparse) on all `num_examples` questions.
6. Saves the combined JSON artifact and four plots to `results/phase2/`.
7. Prints a final summary table: F1 and EM for each config.

**Toggling sparse on/off without editing the script:**  
Phase 1 (`run_phase1.py`) always runs dense — `use_sparse` is ignored there.  
Phase 2 (`run_phase2.py`) hard-codes the three configs internally. To run only dense again (e.g., to re-baseline), simply run `python scripts/run_phase1.py` — the Phase 2 patches do not affect it.

**Time estimates:**

| Hardware | Model | Examples | Estimated time |
|---|---|---|---|
| Laptop (CPU) | Llama-3.2-1B | 2 | ~40 min |
| Laptop (CPU) | Llama-3.2-1B | 10 | ~3–4 hours |
| A100-40GB | Llama-3.1-8B | 100 | ~2–3 hours |

> On the A100 with 100 examples: if Phase 1 results exist, only B5 + B6 run fresh.

---

## Running on a GPU / Cloud Machine

Only **two lines** in `config.py` need to be changed. Everything else (GPU placement, memory management) is already handled automatically by `device_map="auto"`.

### Change 1 — Switch to the full model

In `config.py`, swap the model name:

```python
# Before (local testing):
model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

# After (GPU / A100):
model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
```

### Change 2 — Scale up the number of examples

```python
# Before:
num_examples: int = 10

# After (full Phase 1 run):
num_examples: int = 100
```

### GPU memory requirements

| Model | Precision | VRAM required |
|---|---|---|
| `Llama-3.2-1B-Instruct` | bfloat16 | ~2.5 GB — runs on any GPU or CPU |
| `Llama-3.1-8B-Instruct` | bfloat16 | ~16 GB — requires A100-40GB or equivalent |

### Running on a remote server (SSH / SLURM)

Transfer the project folder including the `data/` directory (already downloaded) to avoid re-downloading on the server. Then:

```bash
cd SURP/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Copy your .env file or export HF_TOKEN directly:
export HF_TOKEN=hf_your_token_here
python scripts/run_phase1.py
```

### Running on Google Colab

```python
# In a Colab cell:
!git clone <your-repo-url>
%cd SURP
!pip install -r requirements.txt
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"  # or use Colab Secrets
!python scripts/run_phase1.py
```

---

## Configuration Reference

All parameters live in `config.py` and apply across all phases:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `torch_dtype` | `bfloat16` | Model weight precision |
| `max_hops` | `4` | Maximum IRCoT retrieve→reason cycles per question |
| `max_new_tokens` | `256` | Max tokens generated per reasoning step |
| `retrieval_top_k` | `3` | Passages retrieved per BM25 query |
| `num_examples` | `10` | Questions to evaluate (set to 100+ for real results) |
| `output_dir` | `results/phase1` | Where JSON artifacts and plots are saved |
| `sink_size` | `128` | **(Phase 2)** First N tokens always attended — covers system prompt + question |
| `local_window` | `2048` | **(Phase 2)** Last N tokens always attended — covers current hop's passage + reasoning |
| `hash_budget` | `256` | **(Phase 2)** Tokens randomly selected from old cycles in the middle region. Set to 0 for B5 (Sink+Local only ablation) |
| `use_sparse` | `False` | **(Phase 2)** `False` → dense generation (Phase 1 behaviour). `True` → SPIRE sparse generation |
| `use_attention_retrieval` | `False` | **(Phase 3 placeholder)** When True, replaces BM25 with attention-map-guided retrieval |

---

## Security Notes

- `HF_TOKEN` is loaded from `.env` via `python-dotenv` and accessed as `os.environ["HF_TOKEN"]`. It is never hardcoded anywhere in source files.
- `.env` is listed in `.gitignore` and must never be committed.
- `results/` and `data/` are also gitignored to avoid accidentally committing large model outputs or dataset files.

### `.gitignore` reference

```
.env
__pycache__/
*.pyc
results/
data/
*.pt
*.bin
.ipynb_checkpoints/
```
