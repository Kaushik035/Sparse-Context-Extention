"""Shared configuration for SPIRE experiments."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env variables before any runtime checks for credentials.
load_dotenv()


@dataclass
class SPIREConfig:
    """Configuration values for Phase 1 IRCoT experiments."""

    # Model
    # model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"

    # IRCoT
    max_hops: int = 4
    max_new_tokens: int = 256
    retrieval_top_k: int = 3

    # BM25
    bm25_chunk_size: int = 100

    # Sparse (Phase 2)
    sink_size: int = 128
    local_window: int = 2048
    hash_budget: int = 256
    use_sparse: bool = False

    # Phase 3
    use_attention_retrieval: bool = False
    truncate_context_tokens: int = 0    # 0 = no truncation; set to 4096 for B3 baseline
    attn_retrieval_max_seq: int = 1500  # raise to 4096 on A100 so B7 uses real attention

    # Phase 3 B8 / B9 — dense and hybrid retrieval
    # B8: pure cosine-similarity retrieval with a sentence-transformer model
    # B9: hybrid BM25 + cosine (weighted combination, score-normalised)
    dense_retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_dense_weight: float = 0.5    # weight for cosine score in hybrid (1-w for BM25)

    # Evaluation
    num_examples: int = 1  # quick validation; change to 10/100 for real runs
    output_dir: str = "results/phase1"

    def output_path(self) -> Path:
        """Return the output directory as a Path."""
        return Path(self.output_dir)
