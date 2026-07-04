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
- [Setup and Installation](#setup-and-installation)
- [Running Phase 1](#running-phase-1)
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
| Phase 2 | Add sparse attention (SPIRE) and compare against dense baselines | Planned |
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
│   ├── model_manager.py            # HuggingFace model loading + generation
│   ├── retriever.py                # BM25 retriever
│   ├── ircot_loop.py               # Interleaved retrieval-reasoning loop
│   └── evaluator.py                # F1 / EM evaluation utilities
│
├── scripts/
│   ├── run_phase1.py               # Phase 1 experiment entrypoint
│   ├── run_phase2.py               # Phase 2 (planned)
│   └── run_phase3.py               # Phase 3 (planned)
│
├── data/
│   └── musique/                    # Auto-downloaded on first run
│
└── results/
    └── phase1/                     # JSON artifacts + plots saved here
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
| `sink_size` | `128` | (Phase 2) Always-attended prefix tokens |
| `local_window` | `2048` | (Phase 2) Dense attention window for current hop |
| `hash_budget` | `256` | (Phase 2) Sparse tokens selected from old context |
| `use_sparse` | `False` | (Phase 2) Toggle sparse attention on/off |

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
