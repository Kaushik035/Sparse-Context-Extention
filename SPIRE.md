# SPIRE: Scalable Interleaved Retrieval-Reasoning via Sparse Attention

> **Undergraduate Research Proposal · B.Tech 3rd Year · Summer 2026**
> Topic: Optimizing Context Window Usage in LLMs for Multi-Hop RAG · Timeline: ~8 Weeks

---

## Table of Contents

- [Deliverable 1 — Research Proposal](#deliverable-1--research-proposal)
  - [Problem Statement](#problem-statement)
  - [Core Insight](#core-insight)
  - [Method](#method)
  - [Experimental Setup](#experimental-setup)
  - [Novelty Argument vs IRCoT](#novelty-argument-vs-ircot)
  - [Phased Plan](#phased-plan)
  - [Week-by-Week Timeline](#week-by-week-timeline)
  - [Risks and Mitigations](#risks-and-mitigations)
- [Deliverable 2 — Presentation Outline (15 min)](#deliverable-2--presentation-outline-15-min)
- [Deliverable 3 — Essential Reading List](#deliverable-3--essential-reading-list)

---

## Deliverable 1 — Research Proposal

---

### Problem Statement

LLM context windows are fundamentally limited. Even frontier models cap out at 128K–256K
tokens, and in practice, quality degrades well before that limit due to lost-in-the-middle
effects and quadratic attention cost. This is exactly why **Retrieval-Augmented Generation
(RAG)** has become the dominant paradigm in industry — instead of stuffing everything into the
window, retrieve only what you need and make every token in the window count.

For multi-hop question answering, **IRCoT** (Trivedi et al., 2023) extended RAG into an
iterative process: retrieve a passage, reason about it, use the reasoning to retrieve the
next passage, and repeat. This interleaved retrieve→think loop significantly outperforms
single-shot RAG on 2–4 hop questions.

**The problem: each cycle consumes more of the limited context window.** After each
retrieve→think cycle, the context accumulates:

```
[question] + [retrieved chunk 1] + [reasoning 1] + [retrieved chunk 2] + [reasoning 2] + ...
```

With realistic passage sizes (500–2,000 tokens in production RAG systems), the window fills
up fast. By hop 4–5, accumulated context reaches 10K–20K tokens. Every token in the window
receives **full dense attention** — the question, the fresh evidence, stale evidence from 3
hops ago, and the model's own previous reasoning are all treated equally. This creates three
problems:

1. **Window budget wasted on stale content** — old reasoning tokens that the model already
   distilled into its current chain still occupy window space AND receive full attention
   compute. The limited window is being spent on content that is no longer critical.
2. **Attention dilution** — fresh evidence (most important for the current step) competes
   equally with everything else in the window. As cycles pile up, the model loses focus.
3. **Ceiling on useful hops** — both effects compound, creating a practical ceiling on how
   many retrieval hops are useful. Beyond this ceiling, adding more evidence to the window
   actually *hurts* accuracy. The window runs out of effective capacity before it runs out
   of tokens.

The core tension: multi-hop reasoning needs MORE evidence (more hops), but the limited
context window can only hold so much before attention becomes ineffective. We need a way to
fit more useful hops into the same window budget.

> **Research question:** Can a structured sparse attention policy — dense over the current
> cycle, sparse over older accumulated cycles — allow IRCoT-style reasoning to fit more
> effective hops into the limited context window, enabling deeper multi-hop chains without
> accuracy degradation?

---

### Core Insight

RAG exists because context windows are limited — so every token in the window should earn
its place. But IRCoT treats all accumulated tokens with equal, dense attention. That's a
waste of the limited window budget.

The key observation: IRCoT already structures context into temporal cycles, and not all
cycles are equally important at any given step. Sparse attention can exploit this structure
to **get more out of the same window**:

- Tokens from the **current** retrieve→think cycle → **dense attention** (fresh, critical —
  this is what the model needs right now)
- Tokens from **older** accumulated cycles → **sparse attention** (still accessible if needed,
  but not consuming full attention budget)
- The **question and system prompt** → **always attended** (the anchor for all reasoning)

The effect: old content stays in the window (no information loss like truncation) but costs
far less attention budget, freeing capacity for the model to focus on fresh evidence. This
means we can fit **more effective hops** into the same limited window — the model can go
deeper in its reasoning chain before accuracy degrades.

This matches exactly what the **sparse-attention-hub** is designed to support with composable
maskers — and it requires **zero model training** in its base form.

---

### Method

#### System Architecture

```
Input question Q, Document corpus D
│
└─► Cycle 1
        Retrieve  : BM25(Q, D)      → passage p₁ (500–2000 tokens)
        Context   : [SYS | Q | p₁]
        Think     : generate r₁ with FULL attention (context is still short)
        Extract   : next query q₁ from r₁
│
└─► Cycle k  (k ≥ 2)
        Retrieve  : BM25(qₖ₋₁, D)  → passage pₖ (500–2000 tokens)
        Context grows: [SYS | Q | p₁ | r₁ | ... | pₖ₋₁ | rₖ₋₁ | pₖ]
        ┌───────────────────────────────────────────────────┐
        │ SPARSE attention over old cycles (p₁..pₖ₋₁, r₁..rₖ₋₁) │
        │ DENSE  attention over current cycle (pₖ)               │
        │ ALWAYS attend to SYS + Q (sink tokens)                 │
        └───────────────────────────────────────────────────┘
        Think     : generate rₖ with SPIRE sparse config
        Extract   : next query qₖ from rₖ, OR emit final answer
```

#### Sparse Attention Configuration (sparse-attention-hub)

```python
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    HashAttentionMaskerConfig,
)

spire_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),       # always attend to: SYS prompt + question
        LocalMaskerConfig(window_size=2048),   # always attend to: current cycle (fresh evidence + reasoning)
        HashAttentionMaskerConfig(budget=256),  # sparse semantic selection from: all older accumulated cycles
    ]
)
```

**Three maskers. No training. Composable and additive.**

The hub combines them as:
> final mask = sink ∪ local window ∪ hash-selected-from-old

#### How Cycle Boundaries Work in Practice

No explicit boundary metadata needs to be passed to the hub. The `LocalMaskerConfig`
automatically covers the last `window_size` tokens (= current cycle's passage + reasoning).
Everything outside the local window falls through to the hash-based sparse selector. This
means SPIRE works with zero modification to how IRCoT builds the prompt — you simply swap
in the sparse adapter.

#### Key Hyperparameters

| Parameter | Role | Starting value | Rationale |
|---|---|---|---|
| `window_size` | Dense coverage of current cycle | 2048 | Covers one full retrieved passage (500–2000 tok) + reasoning (~200 tok) |
| `sink_size` | Always-attended prefix (question) | 128 | System prompt + question typically < 128 tokens |
| `hash budget` | Sparse tokens selected from old context | 256 | ~10–15% of old context; keeps long-range access |

The `window_size` is the main dial:

| window_size too small | window_size too large |
|---|---|
| Model misses the just-retrieved passage | Approaches full attention; no utilization benefit |

In Phase 2, we also explore **adaptive windowing** — dynamically setting `window_size` to the
actual token count of the current cycle, so dense attention always covers exactly one fresh
passage and nothing more.

---

### Experimental Setup

#### Models

| Model | Role | Reason |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Primary | Officially used in hub examples; strong instruction following; 128K window |
| `Qwen2.5-7B-Instruct` | Cross-model check | Different architecture; validates config transfer |

Both run with `torch_dtype=torch.bfloat16` on one A100-40/80GB.

#### Datasets

We use two retrieval scales to stress-test context utilization at different window sizes:

| Dataset | Retrieved chunk size | Context at hop 5 | Role |
|---|---|---|---|
| **MuSiQue** (Trivedi et al., 2022) — full distractor setting | ~100 tok (short paragraphs) + 20 distractor paragraphs per question | ~4K–6K tok | Sanity check; easy scale; validates correctness |
| **MuSiQue-Long** (our variant) | Full Wikipedia article sections (~1K–2K tok per retrieval) via BM25 over a Wikipedia dump | ~12K–20K tok at hop 5 | **Primary evaluation** — realistic scale where sparse attention matters |
| **2WikiMultihopQA** | Same long-chunk setup | ~10K–18K tok | OOD generalization |
| **RULER 4K/8K/16K** | Synthetic | Configurable | Fast validation during sparse config tuning |

> **MuSiQue-Long construction:** For each MuSiQue question, instead of using the provided
> ~100-token paragraphs, retrieve the top-1 passage from a full Wikipedia paragraph corpus
> using BM25. This gives realistic chunk sizes (500–2000 tokens) that stress the context
> window. Gold paragraph titles from MuSiQue are used to validate retrieval quality.
>
> This is a one-time preprocessing step, not a new dataset — just a longer-chunk retrieval
> corpus paired with the same MuSiQue questions and evaluation.

#### Baselines

| ID | Name | Description |
|---|---|---|
| B1 | Retrieve-Once RAG | Single retrieval, full generation — lower bound on multi-hop |
| B2 | IRCoT-Dense | IRCoT with full attention — **primary comparison** |
| B3 | IRCoT-Truncate | IRCoT but truncate context to last 4K tokens (naïve compression) |
| B4 | IRCoT-Summarize | IRCoT but after each cycle, summarize all prior cycles into a short paragraph and drop originals |
| B5 | SPIRE-Sink+Local | Sparse but no hash selection (ablation — tests if local window alone suffices) |
| B6 | SPIRE-Full | Ours: sink + local window + hash sparse over old cycles |
| B7 | SPIRE + Attention Retrieval | Ours + attention-guided retrieval replacing BM25 at each hop |
| B8 | Cosine Retrieval | Dense IRCoT with BM25 replaced by cosine-similarity retrieval (`sentence-transformers/all-MiniLM-L6-v2`). Tests whether semantic embeddings improve passage selection over keyword overlap |
| B9 | Hybrid (BM25 + Cosine) | Dense IRCoT with min-max fused BM25 + cosine retriever (equal weight). Tests whether combining keyword and semantic signals is complementary |

#### Metrics

| Metric | What It Measures |
|---|---|
| Answer F1 / EM | Task accuracy (standard) |
| F1 per hop depth (1/2/3/4) | Where does each method degrade? — **the key figure** |
| Supporting fact recall per hop | Does the model actually use evidence from earlier hops? |
| KV cache memory (GB) at hop k | Memory cost as context grows |
| Prefill time (ms) at hop k | Inference latency as context grows |
| Effective sparsity (%) | Percentage of attention entries skipped |
| Total tokens consumed | End-to-end cost across all hops |

**The key figure to produce:**

```
F1
 │  ● Dense IRCoT (B2)
 │  ■ SPIRE+Attn (B7, ours)
 │  ▲ IRCoT-Truncate (B3)
 │  ◆ Cosine (B8)
 │  ✦ Hybrid BM25+Cosine (B9)
 │
 ●■◆✦▲
 │      ●■◆✦▲
 │          ●■ ◆✦ ▲
 │            ●   ■  ◆✦  ▲     ← truncation collapses; SPIRE degrades gracefully
 │                ●    ■
 └────────────────────────────
   1   2   3   4   5   hop depth  (on MuSiQue-Long)

  ● Dense IRCoT   ■ SPIRE+Attn (ours)   ▲ IRCoT-Truncate   ◆ Cosine   ✦ Hybrid
```

If SPIRE maintains higher F1 at deeper hops where dense IRCoT and truncation degrade — that
is the paper. The story: **sparse attention lets multi-hop RAG fit more productive hops into
the limited context window — old content stays accessible without wasting attention budget,
unlike dense (wastes budget) or truncation (loses information).** B8/B9 isolate whether the
retrieval signal (semantic vs keyword) independently explains any gap, separate from the
attention mechanism used during generation.

---

### Novelty Argument vs IRCoT

| Dimension | IRCoT (Trivedi et al., 2023) | SPIRE (this work) |
|---|---|---|
| Retrieval | Interleaved (BM25 on CoT) | Interleaved (same in Phase 1–2; attention-guided in Phase 3) |
| Reasoning | CoT over full accumulated context | CoT with structured sparse attention over old cycles |
| Attention over old cycles | Full (dense, uniform) | Sparse (hash-selected, recency-aware) |
| Context window utilization | Poor — all content treated equally, window fills up fast | Structured — fresh evidence prioritized, old content costs less budget |
| Scales with hop depth? | Degrades — window budget exhausted, dilution | Better — sparse keeps old accessible at low budget cost |
| Retrieval signal | Static (BM25 on generated text) | Phase 3: attention-guided (what the model is looking for) |
| Requires training? | No | No (Phase 1–2); optional lightweight tuning (Phase 3) |

**In one sentence:** IRCoT tells us *what to retrieve and when*, but wastes the limited context
window by giving equal attention to all accumulated content. SPIRE fixes this by applying
structured sparse attention that prioritizes fresh evidence, letting the model fit more
productive reasoning hops into the same window.

---

### Phased Plan

#### Phase 1 — Reproduce, Scale, and Profile (Weeks 1–3)

**Goal:** Build the IRCoT loop, scale it to realistic chunk sizes, and measure where context
utilization breaks down.

**Week 1:**
- Implement the IRCoT loop in ~150 lines of Python using `ModelAdapterHF` in dense mode
- Use `rank_bm25` for retrieval over MuSiQue's provided paragraphs (short chunks)
- Run 1-hop through 4-hop on 200 MuSiQue dev examples
- Log: F1 per hop, KV memory per hop, prefill time per hop

```python
# Dense baseline — pass None for sparse config
adapter = ModelAdapterHF(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sparse_attention_config=None,   # full attention
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)
```

**Week 2:**
- Build the **MuSiQue-Long** variant: replace short paragraphs with full Wikipedia sections
  retrieved by BM25 (chunk sizes ~500–2000 tokens)
- Re-run the same IRCoT loop with long chunks
- Profile: at which hop depth does F1 start dropping? How large is the context at that point?

**Week 3:**
- Run all baselines that don't need sparse attention: B1 (retrieve-once), B2 (IRCoT-Dense),
  B3 (IRCoT-Truncate), B4 (IRCoT-Summarize)
- Produce the "scaling wall" analysis: F1 vs hop depth, memory vs hop depth, latency vs
  hop depth — across all dense baselines

**Deliverable to professor:** Graphs showing (a) context utilization degrades at deeper hops
with long chunks; (b) truncation and summarization lose information; (c) the gap SPIRE
needs to fill is clear.

---

#### Phase 2 — Sparse Attention Integration (Weeks 4–5)

**Goal:** Add sparse attention, tune configs, show SPIRE utilizes the window better.

**Week 4:**
- Swap `sparse_attention_config=None` for `spire_config`
- Sanity check: single-hop accuracy is within 1–2 F1 points of dense
- Sweep configurations:
  - `window_size` over {1024, 2048, 3072}
  - `sink_size` over {64, 128, 256}
  - `hash budget` over {128, 256, 512}
- Use RULER 4K/8K passkey as fast validation (should get near 100%)

```python
# Use the hub's BenchmarkExecutor for the sweep
from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig

executor = BenchmarkExecutor(gpu_ids=[0], max_concurrent_runs=1)
results = executor.run_benchmark_matrix(
    model_names=["meta-llama/Llama-3.1-8B-Instruct"],
    sparse_attention_configs=[
        ("dense",       None),
        ("spire_w1024", spire_config_w1024),
        ("spire_w2048", spire_config_w2048),
    ],
    benchmark_configs=[
        BenchmarkConfig(benchmark_name="ruler", subsets=["4096", "8192"]),
    ]
)
```

**Week 5:**
- Implement **adaptive windowing**: dynamically set `window_size` = len(current_cycle_tokens)
  so dense attention always covers exactly the latest passage + reasoning, no more
- Run full evaluation: all baselines (B1–B7) on MuSiQue-Long dev (all hop depths)
- Produce: main results table, accuracy-vs-hop-depth figure, memory-vs-hop-depth figure,
  Pareto curves (accuracy vs tokens consumed)

**Deliverable to professor:** Best sparse config chosen; main results showing SPIRE maintains
accuracy at deeper hops; Pareto curves showing better accuracy-per-token.

---

#### Phase 3 — Attention-Guided Retrieval (Weeks 6–7)

**Goal:** Replace BM25 with attention-guided retrieval at each cycle, connecting directly to
AttentionRetriever's insight that attention maps are retrieval signals.

**The idea:** After each reasoning step, instead of running BM25 on the generated text (what
IRCoT does), extract the attention map from the model's last forward pass. Find which
document regions the model attended to most heavily during reasoning — these indicate what
evidence the model was "looking for." Retrieve chunks adjacent to or entity-linked to those
high-attention regions.

**Why this is better than BM25:** BM25 matches surface-level keywords from the CoT text.
Attention-guided retrieval captures what the model *semantically needs* — it might attend
heavily to a date or entity name, indicating it needs more context about that entity, even
if the generated CoT text doesn't mention it explicitly.

**Implementation sketch:**
```python
# After generating reasoning rₖ, extract attention from last forward pass
# Use AttentionRetriever's scoring: max cross-attention per sentence, averaged over heads
attn_maps = model.get_last_attention()  # shape: (layers, heads, seq, seq)

# Score each paragraph in the corpus by attention received from reasoning tokens
# Use only high-retrieval-accuracy layers (layers in the second half of the network,
# following AttentionRetriever's finding)
selected_layers = [20, 22, 24, 26]  # for Llama-3.1-8B (32 layers total)
para_scores = compute_attention_retrieval_scores(attn_maps, selected_layers, corpus)

# Retrieve top-k paragraphs by attention score
next_passage = corpus[para_scores.argmax()]
```

**Week 6:** Implement attention extraction + scoring; validate that attention-selected passages
overlap with BM25-selected passages (sanity check).

**Week 7:** Run full evaluation with attention-guided retrieval; compare SPIRE+BM25 vs
SPIRE+attention-guided; analyze cases where they diverge.

**Deliverable to professor:** Analysis showing attention-guided retrieval improves multi-hop
accuracy by finding evidence that BM25 misses; complete results for all methods.

> **Note:** Phase 3 is the strongest novelty contribution. IRCoT uses BM25.
> AttentionRetriever uses attention for retrieval but not in an interleaved loop.
> SPIRE Phase 3 combines both: interleaved retrieval where the retrieval signal comes from
> the model's own attention during reasoning. This is new.

---

#### Phase 4 — Stretch Goals (Week 8, only if earlier phases are solid)

> Do not start these until Phase 2 is producing clean results.

1. **Attention analysis figure** — plot average attention density over old-cycle tokens vs
   new-cycle tokens across model layers. This is a strong analytical figure that shows
   *why* sparse attention works: the model naturally focuses on recent content, and SPIRE
   simply formalizes that pattern.

2. **Evidence deduplication filter** — before appending a new retrieved passage, check cosine
   similarity to all prior retrieved passages; skip if above a threshold. Two-line
   pre-processing step; meaningful ablation.

3. **Cross-model validation** — run best SPIRE config on Qwen2.5-7B-Instruct to show the
   approach transfers across model families.

4. **Fine-tuning exploration** — take a small model (Llama-3.2-3B) and fine-tune on IRCoT
   trajectories generated with SPIRE's sparse config. Does a model trained with
   cycle-aware sparsity learn to reason more effectively within the sparse budget?

---

### Week-by-Week Timeline

| Week | Focus | Deliverable to Professor |
|---|---|---|
| 1 | Environment setup, IRCoT loop on MuSiQue (short chunks) | Demo: IRCoT loop running on 10 examples |
| 2 | Build MuSiQue-Long (long chunks), profile context growth | Graphs: "context utilization degrades at scale" |
| 3 | Run all dense baselines (B1–B4), document the scaling wall | Full baseline comparison; clear problem statement validated |
| 4 | Sparse attention integration, config sweep, RULER validation | Demo: SPIRE running; sanity check passed |
| 5 | Adaptive windowing, full evaluation on MuSiQue-Long | Main results table + key figures (F1 vs hop depth) |
| 6 | Attention-guided retrieval implementation + sanity checks | Demo: attention-based retrieval selecting passages |
| 7 | Full evaluation with attention-guided retrieval | Complete results; comparison BM25 vs attention-guided |
| 8 | Analysis, cross-model, paper writing | Draft short paper or workshop submission |

---

### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dense IRCoT does NOT degrade even with long chunks | Medium | If context at hop 5 is ~15K tokens and Llama-3.1-8B still handles it fine, extend to 8+ hops or use even longer chunks (full articles). If no degradation at all, the analysis showing "context utilization holds up to X tokens" is itself a publishable finding. |
| Sparse attention collapses accuracy at hop 1 | Low | Increase `window_size` and `sink_size`; start with local+sink-only (B5) before adding hash; verify on RULER first |
| Hash-based masker is unstable across runs | Medium | Use `LocalMaskerConfig`-only first (B5); confirm local window does most of the work; add hash after as incremental improvement |
| MuSiQue-Long retrieval quality is poor (BM25 doesn't find gold paragraphs) | Medium | Validate retrieval recall in Week 2 before proceeding; if recall is too low, fall back to providing gold + distractor paragraphs from MuSiQue but using the full Wikipedia text (longer version of each paragraph) |
| Attention-guided retrieval (Phase 3) doesn't outperform BM25 | Medium | Phase 3 is additive — Phase 2 results stand on their own. Even a negative result ("attention-guided retrieval does not help beyond BM25 in the interleaved setting") is informative and publishable. |
| Hub compatibility issue with chosen model | Low | Llama-3.1-8B-Instruct is the model used in hub's own examples; start there |

**Minimum viable outcome (backup):** Even if sparse attention shows modest improvements, a
careful empirical study showing (a) how context utilization degrades in IRCoT-style loops as
a function of hop depth and chunk size, (b) which sparse patterns preserve multi-hop
reasoning and which don't, and (c) where the accuracy-efficiency frontier lies — is itself a
valuable analytical contribution for a workshop paper.

---

---

## Deliverable 2 — Presentation Outline (15 min)

**Format:** 10 min background · 5 min proposal · Q&A

---

### Slide 1 — Title

> **SPIRE: Scalable Interleaved Retrieval-Reasoning via Sparse Attention**
>
> *[Your name] · Advised by [Prof. name] · [Institution] · June 15, 2026*

Clean title slide — no bullet points. Just title, name, date.

---

### Slide 2 — The Central Question

*(Big font, centered, nothing else on the slide)*

> **"Context windows are limited.
> How do we fit deeper reasoning into them?"**

*Let this sit for 5 seconds before speaking.*
*Say: "RAG solved the problem of getting relevant information into the window. But for
multi-hop reasoning, each hop adds more to the window, and it fills up fast. I'm going to
show you how sparse attention lets us fit more useful reasoning into the same limited space."*

---

### Slide 3 — Why Context Windows Are the Bottleneck

**Header:** *Limited windows → RAG → but RAG has its own scaling problem*

Three points:

- **Context windows are limited** — even frontier models cap at 128K–256K tokens; quality
  degrades well before that limit (lost-in-the-middle, attention dilution)
- **RAG is the industry answer** — instead of stuffing everything in, retrieve only what's
  relevant. Every token in the window should earn its place. This is why RAG is everywhere
  in production.
- **But multi-hop RAG fills the window fast** — interleaved retrieval (IRCoT) adds passages
  and reasoning at each hop. With realistic chunk sizes, the window fills up by hop 4–5,
  and attention spreads thin across all the accumulated content

*Visual: a context window bar filling up across 5 hops. Early hops have room; later hops
are cramped. Label: "Each hop adds ~2K–4K tokens to the window."*

---

### Slide 4 — Why Multi-Hop Reasoning Is the Pressure Point

**Header:** *Each hop consumes window budget — and dense attention wastes it*

- Multi-hop QA requires iterative evidence gathering:
  *"Who wrote the song in the film that starred the actor born in X city?"*
- Each step: retrieve → reason → retrieve again based on what you learned
- Context grows with every cycle: p₁ → r₁ → p₂ → r₂ → ...
- **The waste:** old reasoning from hop 1 still gets full attention at hop 5, even though
  the model already incorporated that information. Dense attention treats all accumulated
  tokens equally — wasting limited window budget on stale content.
- **Why I chose this:** concrete, measurable, and the failure mode (deeper hops → worse
  accuracy) is directly caused by how attention is spent inside the limited window

*Visual: a 4-step hop chain diagram. At each step, the context window bar grows. Color-code:
fresh evidence (bright green) vs old evidence (faded gray). Arrow: "attention budget spread
equally across ALL of this."*

---

### Slide 5 — Paper Explored #1: IRCoT *(the foundation)*

**Header:** *IRCoT — Interleaved retrieval works, but doesn't manage context growth*

- **Trivedi et al. (2023)** | arXiv:2212.10509
- Key idea: use each CoT reasoning step as the next retrieval query; interleave retrieval
  and reasoning
- Result: substantial gains on 2–4 hop QA over retrieve-once RAG
- **The gap it leaves:** full dense attention over ALL accumulated passages + reasoning at
  every step; no mechanism to manage context window utilization as cycles pile up
- **Connection to SPIRE:** SPIRE keeps the IRCoT loop identical but replaces the attention
  mechanism — same retrieval logic, smarter attention allocation

*Visual: the IRCoT loop diagram. Annotate: "context grows here — all tokens get equal
attention, even stale ones."*

---

### Slide 6 — Papers Explored #2: AttentionRetriever + APE

**Header:** *Attention as a signal; not all context needs equal treatment*

**AttentionRetriever (Fu et al., 2026 · arXiv:2602.12278)**
- Attention maps in pretrained LLMs are strong retrieval signals — no training needed
- Specific layers (second half of the network) achieve high retrieval accuracy
- Entity graphs expand retrieval scope beyond directly matched chunks
- *Lesson: what the model attends to tells us what evidence it needs — we can use this
  for smarter retrieval in later phases*

**APE — Adaptive Parallel Encoding (Yang et al., ICLR 2025 · arXiv:2502.05431)**
- Encodes multiple context chunks independently in parallel; merges KV caches
- 4.5× speedup via pre-cached KV states
- Shows that not all context needs joint dense attention — fragments can be composed
- *Lesson: context can be treated non-uniformly without breaking the model*

*Two-column layout. One takeaway bullet each.*

---

### Slide 7 — Papers Explored #3: RLM + Sparse Attention Hub

**Header:** *Decompose the context; control what you attend to*

**RLM — Recursive Language Models (Zhang et al., 2026 · arXiv:2512.24601)**
- Treats the LLM as a REPL: writes code to slice the input and recursively sub-calls
  itself on each slice
- Handles 10M+ tokens by never attending over all of them at once
- *Lesson: structured decomposition of context into sub-problems is a viable strategy
  for utilizing large context windows*

**Sparse Attention Hub (Skylight / Berkeley)**
- Framework that patches any HuggingFace model with configurable sparse attention
- Composable maskers: Sink tokens, Local window, HashAttention (ICML 2025)
- Built-in RULER, LongBench, InfiniteBench benchmarks
- *This is the tool I will use — plug-and-play sparse attention, no custom implementation*

*Side-by-side. Emphasize that the hub makes experimentation practical.*

---

### Slide 8 — How I Got to the Final Idea

**Header:** *Two observations that click together*

**Observation 1** *(from IRCoT + RLM)*
Multi-hop RAG fills the limited context window with a natural temporal structure: **current
cycle (critical)** vs **older cycles (supporting)**. But dense attention ignores this structure
and treats everything equally.

**Observation 2** *(from AttentionRetriever + APE + the hub)*
Not all context needs equal attention budget. AttentionRetriever showed that attention
naturally concentrates on relevant content. APE showed context fragments can be treated
independently. The sparse-attention-hub makes this configurable.

**The Combination:**

> Keep old context in the window (no information loss) but spend attention budget wisely:
> **Dense attention** on the current retrieve→think cycle.
> **Sparse attention** on all prior accumulated cycles.
> Result: more effective hops fit into the same limited window.

*Visual: a before/after attention matrix diagram.*
- Before (IRCoT-Dense): full red matrix — every token gets equal attention budget
- After (SPIRE): dense block at the bottom-right (current cycle) + sparse dots elsewhere
  — budget concentrated where it matters

---

### Slides 9–10 — The Proposal

#### Slide 9 — SPIRE: The Method

**Header:** *Three phases, each building on the last*

**Phase 1 (Weeks 1–3):** Reproduce IRCoT, scale to realistic chunk sizes, measure
where context utilization breaks down. Establish the problem empirically.

**Phase 2 (Weeks 4–5):** Add sparse attention via the hub. Three maskers:
  - Sink (128 tokens) = system prompt + question — always attended
  - Local window (adaptive) = current cycle — dense
  - Hash sparse (budget=256) = everything older — sparse

**Phase 3 (Weeks 6–7):** Replace BM25 with attention-guided retrieval — use the model's
own attention maps to decide what to retrieve next, connecting AttentionRetriever's
insight directly into the interleaved loop.

```python
spire_config = ResearchAttentionConfig(
    masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=2048),
        HashAttentionMaskerConfig(budget=256),
    ]
)
```

---

#### Slide 10 — Experiments and Expected Results

**Header:** *What I will measure, and what I expect to find*

| | Details |
|---|---|
| **Primary dataset** | MuSiQue-Long (realistic chunk sizes, 10K–20K context at hop 5) |
| **OOD dataset** | 2WikiMultihopQA |
| **Baselines** | Retrieve-Once (B1) · IRCoT-Dense (B2) · IRCoT-Truncate (B3) · SPIRE-Full (B6) · SPIRE+Attention (B7) · Cosine (B8) · Hybrid BM25+Cosine (B9) |
| **Key metric** | F1 at each hop depth: 1 / 2 / 3 / 4 |
| **Secondary** | KV memory, prefill latency, total tokens consumed |

**Expected finding:**

```
F1
 │ ●■◆✦▲
 │       ●■◆✦ ▲
 │           ●■  ◆✦  ▲
 │             ●   ■   ◆✦   ▲   ← truncation drops; SPIRE holds; B8/B9 isolate retrieval effect
 │                 ●     ■
 └──────────────────────────────
   1   2   3   4   5   hop depth

  ● Dense IRCoT (B2)   ■ SPIRE+Attn (B7)   ▲ Truncate (B3)   ◆ Cosine (B8)   ✦ Hybrid (B9)
```

**The story:** Sparse attention optimizes how the limited context window is spent — old content
stays accessible (unlike truncation) but doesn't waste attention budget (unlike dense). B8/B9
isolate whether the *retrieval signal* (semantic vs keyword) independently drives accuracy,
decoupled from the *attention mechanism* used during generation.

---

### Slide 11 — Q&A

**Questions?**

*[Your name] · [email] · github.com/[your-handle]*

**Anticipated questions:**

| Question | Short answer |
|---|---|
| "Why not just truncate old context?" | Truncation is lossy and irreversible; sparse keeps all tokens accessible at low cost |
| "Why not summarize old context?" | Summarization is a baseline (B4); it's lossy too and adds generation cost per cycle |
| "Is the sparse config fixed or learned?" | Fixed in Phase 2; adaptive windowing already helps; fine-tuning is a stretch goal |
| "What if the accuracy gap doesn't appear?" | We scale chunk sizes until it does; worst case, the analysis is publishable |
| "How is Phase 3 different from AttentionRetriever?" | AttentionRetriever retrieves once over a static document; SPIRE retrieves iteratively using attention that evolves with each reasoning step |

---

---

## Deliverable 3 — Essential Reading List

**Only two papers. Read only the listed sections.**

---

### Paper 1 — HashAttention: Semantic Sparsity for Faster Inference

**Authors:** Desai et al. (ICML 2025)
**Link:** https://openreview.net/forum?id=Em2oaXd8Dc

**Why you need it:**
The `HashAttentionMaskerConfig` inside the sparse-attention-hub is a direct implementation of
this paper. You need to understand what it is actually computing — specifically what "semantic
sparsity" means, what the hash bucket size controls, and what the known failure modes are
(missed high-value tokens in certain distributions). Without this, you will tune it blindly.

**Sections to read (not the whole paper):**

| Section | What you get |
|---|---|
| Section 3 — Method | Understand the LSH-based key selection mechanism |
| Section 4.2 — Ablations on budget | Tells you what `budget` value to start with |
| Table 2 | Shows which task types sparse attention hurts most — directly useful for interpreting your own results |

---

### Paper 2 — MuSiQue: Multihop Questions via Single-hop Question Composition

**Authors:** Trivedi et al. (2022)
**Link:** https://arxiv.org/abs/2108.00573

**Why you need it:**
MuSiQue is your primary dataset. You need to understand its construction — specifically how
hop structure is annotated, how hard negatives work, and what "answerable" vs "unanswerable"
splits mean — to set up evaluation correctly and interpret per-hop results. You cannot rely on
HuggingFace loading alone without knowing the data format.

**Sections to read (not the whole paper):**

| Section | What you get |
|---|---|
| Section 3 — Dataset Construction | Understand what 2-hop, 3-hop, 4-hop mean structurally |
| Section 4 — Experiments | Understand baseline numbers so you know where to aim |
| Appendix A — Data Format | Understand the `paragraphs` and `question_decomposition` fields you will use for retrieval |

---

> **Everything else you need** is already in the papers you have read, the
> [sparse-attention-hub README](https://github.com/skylight-org/sparse-attention-hub),
> and the `rank_bm25` documentation. Do not read more — start implementing.

---

*Last updated: June 14, 2026*