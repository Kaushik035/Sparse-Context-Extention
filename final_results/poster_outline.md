# Poster Outline

## 1. Problem Statement
- Multi-hop RAG grows the context window at every retrieve-reason step.
- Dense attention gives stale evidence the same budget as the current hop.
- BM25 retrieval can miss semantically relevant passages when wording changes.

## 2. Background
- Benchmark: MuSiQue v1.0 multi-hop QA.
- Baseline pipeline: IRCoT with BM25 passage retrieval and LLM reasoning.
- Metrics: SQuAD-style token F1 and exact match, plus context-token growth per hop.
- Sources used in the implementation: Hugging Face datasets first, then official MuSiQue Google Drive fallback if needed.

## 3. Proposal
- SPIRE allocates attention budget by region: sink, local window, and sparse middle.
- The prompt was tightened to a strict protocol: one concise reasoning step, then exactly one Search Query or Final Answer line.
- The final reported runs used Llama-3.1-8B-Instruct for stronger instruction following; the repo still keeps a 1B local-testing option.
- Implemented baselines:
  - B1: retrieve-once
  - B2: dense IRCoT
  - B3: IRCoT with truncation
  - B5: sink + local sparse attention
  - B6: full SPIRE sparse attention
  - B7: SPIRE + attention-guided retrieval
 - B4 is defined in the proposal as summarization, but it is not part of the saved benchmark artifacts.
 - B8 and B9 exist in the codebase as extensions, but they are not part of the saved comparison figures used for the poster.

## 4. Results
- The saved benchmark set is strongest on the dense / sparse-attention comparison because the validation slice in the recorded run is effectively 2-hop, so hop-wise curves collapse to a single measured point.
- Dense IRCoT and full SPIRE were tied at the top on this run: F1 = 0.3811, EM = 0.26.
- Sink + local sparse attention was close behind: F1 = 0.3588, EM = 0.25.
- Truncation hurt quality: F1 = 0.3268, EM = 0.23.
- Retrieve-once was substantially worse: F1 = 0.2117, EM = 0.11.
- Attention-guided retrieval underperformed in this preliminary run: F1 = 0.1896, EM = 0.13.
- Context length still grew with hop depth, reaching roughly 1.35k tokens by hop 4 in the saved Phase 1/2 plots.

## 5. Conclusion
- The main result is that SPIRE changes *where* the model spends attention, not just how much context it sees. That is why full SPIRE can match dense IRCoT while using a more selective attention pattern.
- Truncation is weaker because it destroys older evidence. If a later hop needs an entity from hop 1, truncation removes that information permanently, so performance drops even when the prompt still looks long enough.
- Sink + local sparse attention stays competitive because the question, system prompt, and current evidence remain fully accessible. The missing middle budget mostly affects older passages that are still available but no longer dominate reasoning.
- Retrieve-once is a lower bound because it cannot recover from a bad first retrieval. In multi-hop QA, that matters because the answer often requires chaining through several passages.
- Attention-guided retrieval is promising in theory, but the run is still limited by sequence-length guards, noisy attention scores, and the fact that reasoning traces are short on this benchmark slice.
- Prompt tightening helped because it forced the model into a stable IRCoT protocol: one reasoning step, then one explicit search query or final answer. That reduces drift, makes retrieval queries shorter, and prevents the loop from feeding back verbose chain-of-thought text.
- The larger 8B model is configured in the repo because a stronger instruction-following model should, in principle, improve protocol adherence and answer extraction. That is a model-capacity hypothesis, not a measured result in the saved runs, so it should be phrased as an implementation choice or future experiment unless you rerun the benchmark.
- Best next steps are a larger evaluation set, a proper B4 summarization run if you want the proposal to be complete, and a rerun with the 8B model plus any prompt variants you want to compare fairly.

## Suggested Poster Figures
- Use `f1_em_comparison.svg` as the main results chart.
- Use `context_growth.svg` as the supporting context-utilization chart.
- If space is tight, combine them into one two-panel figure.

## Data and Model Sources
- MuSiQue v1.0 from Hugging Face datasets when available.
- Official MuSiQue Google Drive archive fallback from the Stony Brook NLP download script.
- LLM: Meta-Llama-3.1-8B-Instruct from Hugging Face.
- Dense semantic retriever code path: sentence-transformers/all-MiniLM-L6-v2.