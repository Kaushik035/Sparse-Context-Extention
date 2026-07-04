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

    # Sparse (Phase 2 placeholders)
    sink_size: int = 128
    local_window: int = 2048
    hash_budget: int = 256
    use_sparse: bool = False

    # Evaluation
    num_examples: int = 10 # change to 100 later
    output_dir: str = "results/phase1"

    def output_path(self) -> Path:
        """Return the output directory as a Path."""
        return Path(self.output_dir)
