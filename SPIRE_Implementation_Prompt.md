# SPIRE Implementation Prompt — For Copilot / Claude Code

> **Give this entire file to your AI coding assistant. It contains everything needed to implement SPIRE phase by phase.**

---

## Project Context

I am implementing **SPIRE (Sparse Interleaved Retrieval-Reasoning)** — a research project that improves multi-hop question answering by applying sparse attention during interleaved retrieval-reasoning (IRCoT) loops.

**Current directory:** `SURP/`
**Existing file:** `SURP/SPIRE.md` (research proposal — read it for full context)

---

## What I Need You To Do

Implement this project **phase by phase**. After each phase, I will verify everything works before moving to the next. Between phases:
- Do NOT rewrite existing code — only **patch** what's needed
- Create new files/folders **only when required**
- Keep code **modular** — use classes and functions, no script-style spaghetti
- Every function should have a docstring

---

## Resources & API Keys I Have Access To

- **GPU:** A100 (40GB or 80GB) — available via lab/Colab/cloud
- **HuggingFace Token:** I will provide `HF_TOKEN` for gated model access (Llama-3.1-8B-Instruct)
- **No paid API keys needed** — we run everything locally on GPU

---

## Tech Stack

| Need | Library | Install |
|---|---|---|
| LLM | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace `transformers` | `pip install transformers accelerate` |
| Tokenizer + Generation | HuggingFace `transformers` | already included above |
| Dataset | MuSiQue from HuggingFace `datasets` | `pip install datasets` |
| BM25 Retrieval | `rank_bm25` | `pip install rank-bm25` |
| Evaluation (F1/EM) | HuggingFace `evaluate` with `squad` metric | `pip install evaluate` |
| Sparse Attention | `sparse-attention-hub` from GitHub | `pip install git+https://github.com/facebookresearch/sparse-attention.git` (check exact URL below) |
| Torch | PyTorch with CUDA | `pip install torch` |
| Config | YAML or dataclasses | built-in |
| Logging | Python `logging` + `json` for results | built-in |
| Plotting | `matplotlib` | `pip install matplotlib` |

**Important — Sparse Attention Hub:**
- GitHub: Search for `sparse-attention-hub` or `skylight-research/sparse-attention`
- If the exact package name differs, search PyPI/GitHub for "sparse attention hub masker hashattention huggingface"
- If the hub is unavailable or hard to install, we can **implement sparse attention manually** using a custom attention mask — it's just a boolean mask applied before softmax. I'll explain the fallback below.

**Fallback for sparse attention (if hub not available):**
```python
# Manual sparse mask — works with any HuggingFace model
# Create a custom attention mask of shape (seq_len, seq_len)
# where mask[i][j] = True means token i CAN attend to token j

def build_spire_mask(seq_len, sink_size=128, local_window=2048, hash_budget=256):
    """Build SPIRE sparse attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        # Sink: always attend to first sink_size tokens
        mask[i, :min(sink_size, seq_len)] = True
        # Local: attend to last local_window tokens
        start = max(0, i - local_window + 1)
        mask[i, start:i+1] = True
    # Hash: for tokens outside local window, select top-k by cosine similarity
    # (implemented separately using key states)
    return mask
```

This can be passed as `attention_mask` to `model.generate()` or `model.forward()`. It won't give the FlashAttention speedup of the hub, but it will produce the **correct sparse attention behavior** for our experiments.

---

## Folder Structure (Final — built incrementally)

```
SURP/
├── SPIRE.md                    # existing research proposal
├── .env                        # HF_TOKEN lives here (never commit)
├── .gitignore                  # ignores .env, __pycache__, results/
├── requirements.txt            # created in Phase 1
├── config.py                   # shared config (dataclasses, loads .env)
├── data/
│   └── musique/                # auto-downloaded by datasets library
├── src/
│   ├── __init__.py
│   ├── model_manager.py        # Phase 1: load/manage the LLM
│   ├── retriever.py            # Phase 1: BM25 retriever
│   ├── ircot_loop.py           # Phase 1: the IRCoT loop
│   ├── evaluator.py            # Phase 1: F1/EM evaluation
│   ├── sparse_attention.py     # Phase 2: sparse mask builder (patch)
│   └── attention_retriever.py  # Phase 3: attention-guided retrieval (patch)
├── scripts/
│   ├── run_phase1.py           # Phase 1: run IRCoT + profile
│   ├── run_phase2.py           # Phase 2: run SPIRE + evaluate
│   └── run_phase3.py           # Phase 3: run with attention retrieval
├── results/
│   ├── phase1/                 # F1 scores, profiling data, plots
│   ├── phase2/                 # SPIRE results, comparison plots
│   └── phase3/                 # Attention retrieval results
└── notebooks/                  # optional Jupyter notebooks for analysis
    └── analysis.ipynb
```

**Do NOT create all folders upfront.** Create them as each phase requires.

---

## PHASE 1 — Reproduce IRCoT + Profile (Weeks 1–3)

### Goal
Build a working IRCoT loop, run it on MuSiQue, measure F1 per hop depth, and profile where context growth causes degradation.

### Step 1.1 — Setup

Create `requirements.txt`:
```
torch>=2.0
transformers>=4.40
accelerate
datasets
rank-bm25
evaluate
matplotlib
tqdm
python-dotenv
```

Create `.env` file (this stores credentials — **never commit this**):
```
HF_TOKEN=hf_your_token_here
```

Create `.gitignore`:
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

Create `config.py` with a dataclass:
```python
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file — must be called before accessing any env vars
load_dotenv()

@dataclass
class SPIREConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    
    # IRCoT
    max_hops: int = 4
    max_new_tokens: int = 256
    retrieval_top_k: int = 3
    
    # BM25
    bm25_chunk_size: int = 100  # words per chunk for BM25 indexing
    
    # Sparse (Phase 2 — leave as defaults for now)
    sink_size: int = 128
    local_window: int = 2048
    hash_budget: int = 256
    use_sparse: bool = False  # toggled in Phase 2
    
    # Evaluation
    num_examples: int = 100  # start small, scale up later
    output_dir: str = "results/phase1"
```

### Step 1.2 — Model Manager (`src/model_manager.py`)

Create a class `ModelManager` that:
- Loads the model and tokenizer from HuggingFace
- Handles `torch_dtype=torch.bfloat16`
- Has a `generate(prompt: str, max_new_tokens: int) -> str` method
- Uses the model's chat template for formatting (Llama uses `<|begin_of_text|>` etc.)
- Tracks token counts per generation (for profiling)
- Has a method `get_context_length(prompt: str) -> int` that returns tokenized length

```python
class ModelManager:
    def __init__(self, config: SPIREConfig):
        # Load model and tokenizer
        # Use device_map="auto" for automatic GPU placement
        # Token is auto-loaded from .env via: token=os.environ["HF_TOKEN"]
        pass
    
    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        """Generate response from chat messages. Returns generated text only."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
```

**Important:** Use the chat template properly. Llama-3.1-Instruct uses:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "..."}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
```

### Step 1.3 — Retriever (`src/retriever.py`)

Create a class `BM25Retriever` that:
- Takes a list of passages (strings) and indexes them with BM25
- Has a `retrieve(query: str, top_k: int) -> list[str]` method
- Simple, clean, no over-engineering

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, passages: list[str]):
        """Index passages with BM25."""
        pass
    
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve top-k passages for query."""
        pass
```

### Step 1.4 — IRCoT Loop (`src/ircot_loop.py`)

This is the core. Create a class `IRCoTLoop` that implements the interleaved retrieval-reasoning loop:

```python
class IRCoTLoop:
    def __init__(self, model: ModelManager, retriever: BM25Retriever, config: SPIREConfig):
        pass
    
    def run(self, question: str) -> dict:
        """
        Run IRCoT loop on a question. Returns:
        {
            "question": str,
            "answer": str,
            "num_hops": int,
            "reasoning_chain": [str],  # T1, T2, T3...
            "retrieved_passages": [str],
            "context_tokens_per_hop": [int],  # for profiling
            "total_tokens": int,
        }
        """
        pass
```

**How the loop works:**

```
1. Set system_prompt = "Answer the question by reasoning step by step. 
   When you have the final answer, write 'So the answer is: <answer>'."

2. retrieved_so_far = []
   reasoning_so_far = []
   current_query = question

3. For hop in range(max_hops):
   a. passages = retriever.retrieve(current_query, top_k=3)
   b. retrieved_so_far.extend(passages)
   c. Build the full prompt:
      - System prompt
      - "Question: {question}"
      - "Evidence: {all retrieved passages joined}"
      - "Previous reasoning: {all reasoning steps joined}"
      - "Continue reasoning step by step:"
   d. response = model.generate(prompt)
   e. reasoning_so_far.append(response)
   f. Track context_tokens_per_hop.append(model.count_tokens(full_prompt))
   g. If "the answer is" in response.lower():
      - Extract final answer (text after "the answer is:" or "the answer is")
      - Break
   h. Else:
      - current_query = response  # use reasoning as next retrieval query

4. Return results dict
```

**Extracting the answer:** Use a simple regex:
```python
import re
match = re.search(r"(?:the answer is|answer is)[:\s]*(.+?)(?:\.|$)", response, re.IGNORECASE)
if match:
    answer = match.group(1).strip()
```

### Step 1.5 — Evaluator (`src/evaluator.py`)

```python
class Evaluator:
    def __init__(self):
        pass
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lowercase, remove articles, punctuation, extra whitespace."""
        pass
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """Compute token-level F1."""
        pass
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        """Check exact match after normalization."""
        pass
    
    def evaluate_results(self, results: list[dict], gold_answers: list[str]) -> dict:
        """
        Compute overall F1/EM and F1 per hop depth.
        Returns:
        {
            "overall_f1": float,
            "overall_em": float,
            "f1_by_hops": {2: float, 3: float, 4: float},
            "context_tokens_by_hop": {1: [int], 2: [int], ...},
        }
        """
        pass
```

Use the standard SQuAD normalization:
```python
def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s
```

### Step 1.6 — Dataset Loading

MuSiQue is on HuggingFace:
```python
from datasets import load_dataset

# MuSiQue dataset
dataset = load_dataset("drt/musique", split="validation")
# Each item has: "question", "answer", "paragraphs" (list of dicts with "title", "paragraph_text", "is_supporting")
# "answerable" field — filter to only answerable=True
```

**If `drt/musique` doesn't work**, try:
- `musique` 
- `StonyBrookNLP/musique`
- Or download from: https://github.com/StonyBrookNLP/musique and load locally

Each MuSiQue example has:
- `question`: the multi-hop question
- `answer`: gold answer string
- `paragraphs`: list of paragraph dicts, each with `title`, `paragraph_text`, `is_supporting` (bool)
- The number of hops can be inferred from counting `is_supporting=True` paragraphs (usually 2, 3, or 4)

### Step 1.7 — Run Script (`scripts/run_phase1.py`)

```python
"""Phase 1: Run IRCoT on MuSiQue and profile context utilization."""

def main():
    # 1. Load config
    # 2. Load model (ModelManager)
    # 3. Load dataset (MuSiQue validation, filter answerable, take first N)
    # 4. For each example:
    #    a. Build BM25 index from example's paragraphs
    #    b. Run IRCoT loop
    #    c. Collect result
    # 5. Evaluate (F1 per hop depth)
    # 6. Save results to results/phase1/
    # 7. Plot: F1 vs hop depth, context tokens vs hop depth
    pass
```

### Phase 1 Deliverables (Verify Before Moving On)

- [ ] IRCoT loop runs on 10 examples without crashing
- [ ] F1 scores are reasonable (30–50 F1 on MuSiQue is expected)
- [ ] Context token counts increase with each hop (plot this)
- [ ] Results saved as JSON in `results/phase1/`
- [ ] Plot: F1 per hop depth shows degradation at deeper hops
- [ ] Plot: context token count per hop shows growth

---

## PHASE 2 — Add Sparse Attention (Weeks 4–5)

### Goal
Patch the existing code to use sparse attention during generation. Compare SPIRE vs dense IRCoT.

### What Changes (Minimal Patches)

**1. Create `src/sparse_attention.py`:**

```python
import torch

class SparseAttentionMask:
    """Builds SPIRE-style sparse attention masks."""
    
    def __init__(self, sink_size: int = 128, local_window: int = 2048, hash_budget: int = 256):
        self.sink_size = sink_size
        self.local_window = local_window
        self.hash_budget = hash_budget
    
    def build_mask(self, seq_len: int, key_states: torch.Tensor = None) -> torch.Tensor:
        """
        Build sparse causal attention mask.
        
        Args:
            seq_len: current sequence length
            key_states: optional key states for hash-based selection
        
        Returns:
            mask of shape (seq_len, seq_len), True = attend, False = skip
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            # 1. Sink: always attend to first sink_size tokens
            mask[i, :min(self.sink_size, i + 1)] = True
            
            # 2. Local: attend to last local_window tokens
            local_start = max(0, i - self.local_window + 1)
            mask[i, local_start:i + 1] = True
            
            # 3. Hash: for positions outside sink+local, select by similarity
            # (if key_states provided, use cosine similarity; otherwise random)
            if key_states is not None and i > self.sink_size + self.local_window:
                # Find tokens in range [sink_size, local_start) that aren't already attended
                candidate_range = range(self.sink_size, local_start)
                if len(candidate_range) > self.hash_budget:
                    # Compute cosine similarity between current key and candidate keys
                    current_key = key_states[i]
                    candidate_keys = key_states[list(candidate_range)]
                    similarities = torch.cosine_similarity(
                        current_key.unsqueeze(0), candidate_keys, dim=-1
                    )
                    top_indices = similarities.topk(self.hash_budget).indices
                    for idx in top_indices:
                        mask[i, candidate_range[idx]] = True
                else:
                    # All candidates fit in budget
                    mask[i, self.sink_size:local_start] = True
        
        return mask
```

**NOTE:** The hash selection above uses cosine similarity on key states as a proxy for LSH. This is simpler than true LSH but produces the same effect — selecting semantically similar old tokens. For a research prototype, this is sufficient. If performance is too slow, simplify to random selection from old tokens (still shows the sparse attention pattern works).

**PRACTICAL ALTERNATIVE if custom masks are hard to integrate:**

Instead of modifying the attention mask directly, use a simpler approach: **truncate old context but keep a summary + the question**. This isn't true sparse attention but approximates it:

```python
def build_spire_context(question, old_passages, old_reasoning, current_passage, max_old_tokens=512):
    """Keep question + truncated old context + full current passage."""
    # Always keep question
    # From old passages/reasoning, keep only the last max_old_tokens tokens
    # Keep full current passage
    pass
```

**Use the full sparse mask approach if possible.** Use the truncation approximation only as a last resort.

**2. Patch `src/model_manager.py`:**

Add a method that accepts a custom attention mask:

```python
def generate_with_mask(self, messages, attention_mask=None, max_new_tokens=256):
    """Generate with optional custom attention mask."""
    # If attention_mask provided, pass it to model.generate()
    # HuggingFace transformers accepts attention_mask parameter
    pass
```

**3. Patch `src/ircot_loop.py`:**

Add a flag `use_sparse` that toggles between dense and sparse attention:

```python
# In the loop, before generation:
if self.config.use_sparse:
    mask_builder = SparseAttentionMask(
        sink_size=self.config.sink_size,
        local_window=self.config.local_window,
        hash_budget=self.config.hash_budget,
    )
    mask = mask_builder.build_mask(context_length)
    response = self.model.generate_with_mask(messages, attention_mask=mask)
else:
    response = self.model.generate(messages)
```

**4. Create `scripts/run_phase2.py`:**

```python
"""Phase 2: Compare Dense IRCoT vs SPIRE on MuSiQue."""

def main():
    # 1. Run dense IRCoT (use Phase 1 results if already saved)
    # 2. Run SPIRE (use_sparse=True) on same examples
    # 3. Compare F1 per hop depth
    # 4. Plot both curves on same graph
    # 5. Also run ablations: SPIRE without hash (sink+local only) = B5
    pass
```

### Phase 2 Deliverables

- [ ] Sparse attention mask builds correctly (visualize it for a 1000-token sequence)
- [ ] SPIRE runs on 10 examples without crashing
- [ ] Comparison plot: Dense IRCoT vs SPIRE F1 per hop depth
- [ ] Ablation: SPIRE-Full vs SPIRE-Sink+Local (no hash)
- [ ] Memory profiling: KV cache size comparison
- [ ] Results saved to `results/phase2/`

---

## PHASE 3 — Attention-Guided Retrieval (Weeks 6–7)

### Goal
Replace BM25 with attention-based retrieval. After each reasoning step, extract attention maps and use them to guide the next retrieval.

### What Changes (Patches Only)

**1. Create `src/attention_retriever.py`:**

```python
class AttentionRetriever:
    """Retrieves passages using the model's attention patterns instead of BM25."""
    
    def __init__(self, model: ModelManager, passages: list[str]):
        self.model = model
        self.passages = passages
    
    def retrieve(self, question: str, reasoning_so_far: str, top_k: int = 3) -> list[str]:
        """
        Use attention scores to find relevant passages.
        
        1. Run a forward pass with question + reasoning
        2. Extract attention weights from the last few layers
        3. For each candidate passage, compute an attention-based relevance score
        4. Return top-k passages
        """
        pass
    
    def _extract_attention_scores(self, input_ids, target_layers=None):
        """
        Run forward pass with output_attentions=True.
        Extract attention weights from specified layers.
        
        model outputs have .attentions — tuple of (batch, heads, seq, seq) per layer
        Use layers in the second half of the network (e.g., layers 16-31 for a 32-layer model).
        """
        pass
    
    def _score_passage(self, attention_weights, passage_token_range):
        """
        Score a passage by averaging the max attention weights 
        that query tokens give to tokens in this passage's range.
        """
        pass
```

**How attention extraction works in HuggingFace:**
```python
outputs = model(input_ids, output_attentions=True)
# outputs.attentions is a tuple of length num_layers
# Each element is (batch_size, num_heads, seq_len, seq_len)
# Use the last 8 layers (second half of the network)
attention_maps = outputs.attentions[-8:]  # last 8 layers
```

**Scoring a passage:**
For each passage, identify which token positions it occupies in the input sequence. Then for the attention maps from the last generated tokens (the reasoning), find the maximum attention weight given to any token in that passage. Average across heads and layers. Higher score = more relevant passage.

**2. Patch `src/ircot_loop.py`:**

Add an option to use `AttentionRetriever` instead of `BM25Retriever`:

```python
# In config, add:
use_attention_retrieval: bool = False  # toggled in Phase 3

# In the loop:
if self.config.use_attention_retrieval:
    passages = self.attention_retriever.retrieve(question, reasoning_so_far, top_k=3)
else:
    passages = self.bm25_retriever.retrieve(current_query, top_k=3)
```

**3. Create `scripts/run_phase3.py`:**

Compare all methods:
- B1: Retrieve-Once (single BM25 retrieval)
- B2: IRCoT-Dense (Phase 1)
- B3: IRCoT-Truncate (truncate to last 4K tokens)
- B6: SPIRE-Full (Phase 2)
- B7: SPIRE + Attention Retrieval (Phase 3)
- B8: Cosine Retrieval — BM25 replaced by `sentence-transformers/all-MiniLM-L6-v2` cosine similarity
- B9: Hybrid (BM25 + Cosine) — min-max fused retriever, equal weight by default

### Phase 3 Deliverables

- [ ] Attention extraction works (visualize attention heatmap for one example)
- [ ] Attention-guided retrieval finds relevant passages
- [ ] `src/dense_retriever.py` — `CosineRetriever` (B8) and `HybridRetriever` (B9) with shared `SentenceTransformer` model (loaded once per run)
- [ ] Full comparison table: all 7 baselines vs SPIRE variants
- [ ] Final plots: F1 vs hop depth for all 7 methods on one set of axes
- [ ] Results saved to `results/phase3/`

**New config fields required for B8/B9:**
```python
dense_retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
hybrid_dense_weight: float = 0.5
```
**New dependency required:**
```
sentence-transformers>=2.7
```

---

## Practical Tips

### Running on Limited GPU Memory

If A100-40GB isn't enough for Llama-3.1-8B in bfloat16:
- Use `load_in_4bit=True` with `bitsandbytes` (`pip install bitsandbytes`)
- Or use a smaller model: `meta-llama/Llama-3.2-3B-Instruct` (3B params, fits easily)
- The sparse attention results should transfer across model sizes

### Start Small

- First run everything with `num_examples=10` to verify correctness
- Then scale to `num_examples=100` for real results
- Only go to full dataset (~1000 examples) for final numbers

### If Sparse Mask Integration Is Difficult

The hardest part might be passing custom attention masks through HuggingFace's `generate()`. If this is problematic:

**Option A:** Use `model.forward()` manually in a loop instead of `model.generate()`:
```python
# Manual autoregressive generation with custom mask
for step in range(max_new_tokens):
    outputs = model(input_ids, attention_mask=custom_mask)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    # Extend custom_mask for the new token
```

**Option B:** Use the context-truncation approximation described in Phase 2 above. This is simpler and still demonstrates the core idea (not all context needs full attention).

**Option C:** Use `transformers` model hooks to modify attention at specific layers. Search for `model.register_forward_hook` to intercept and modify attention masks inside the model.

### Logging

Use Python `logging` throughout. Every run should save:
```python
{
    "config": {...},           # full config
    "results": [...],          # per-example results
    "metrics": {...},          # aggregated metrics
    "timestamp": "...",
    "model_name": "...",
    "phase": 1
}
```

Save as `results/phase{N}/run_{timestamp}.json`.

---

## Summary — What To Do Right Now

1. Read `SPIRE.md` in this folder for full research context
2. Create `requirements.txt` and `pip install -r requirements.txt`
3. Create `config.py`
4. Implement Phase 1 files: `src/model_manager.py`, `src/retriever.py`, `src/ircot_loop.py`, `src/evaluator.py`
5. Create `scripts/run_phase1.py` and run on 10 examples
6. Verify F1 scores are reasonable, context grows with hops
7. STOP — tell me the results before Phase 2

**(DONT FORGOT TO CREATE virtual environment before starting work)**

I have alreade done the below HuggingFace token setu till step 3

**HuggingFace token setup:**
1. Get a **Read** token from https://huggingface.co/settings/tokens
2. Accept the Llama-3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (affiliation: Indian Institute of Technology Bombay)
3. Create `.env` file in project root with `HF_TOKEN=hf_your_token_here`
4. Code auto-loads it via `load_dotenv()` in `config.py` — never hardcode tokens anywhere

