"""Phase 1: Run IRCoT on MuSiQue and profile context utilization."""

import sys
from pathlib import Path

# Ensure project root (SURP/) is on sys.path regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import zipfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from config import SPIREConfig
from src.evaluator import Evaluator
from src.ircot_loop import IRCoTLoop
from src.model_manager import ModelManager
from src.retriever import BM25Retriever


def setup_logging() -> None:
    """Configure root logging format for phase runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _download_musique_gdrive(data_dir: Path) -> List[Dict]:
    """Download MuSiQue v1.0 from Google Drive (official source) and parse the validation JSONL."""
    # Google Drive file ID from the official StonyBrookNLP/musique download script.
    gdrive_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
    zip_path = data_dir / "musique_v1.0.zip"
    extract_dir = data_dir / "musique_v1.0"

    if not zip_path.exists():
        logging.info("Downloading MuSiQue v1.0 from Google Drive (~70 MB) ...")
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "'gdown' is required to download MuSiQue. Run: pip install gdown"
            ) from exc
        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
        logging.info("Download complete: %s", zip_path)

    # Extract every time unless we can already find the target file.
    # Avoids hard-coding the subfolder name that the zip creates.
    existing = list(data_dir.rglob("musique_ans_v1.0_dev.jsonl")) + list(data_dir.rglob("*dev*.jsonl"))
    if not existing:
        logging.info("Extracting MuSiQue ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

    # Search recursively — works regardless of what subfolder the zip created.
    candidates = sorted(
        list(data_dir.rglob("musique_ans_v1.0_dev.jsonl"))
        + list(data_dir.rglob("*dev*.jsonl"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a dev JSONL anywhere under {data_dir}. "
            f"Contents: {list(data_dir.rglob('*.jsonl'))}"
        )

    val_file = candidates[0]
    logging.info("Loading validation split from %s", val_file)
    examples: List[Dict] = []
    with val_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_musique_validation() -> List[Dict]:
    """Load MuSiQue validation split — HuggingFace Hub first, GitHub fallback."""
    # Try a broader set of known HF identifiers first.
    hf_candidates = [
        "drt/musique",
        "musique",
        "StonyBrookNLP/musique",
        "datasets-community/musique",
        "alexandrainst/musique",
    ]

    for dataset_name in hf_candidates:
        try:
            from datasets import load_dataset
            logging.info("Trying HuggingFace dataset: %s", dataset_name)
            dataset = load_dataset(dataset_name, split="validation")
            return list(dataset)
        except Exception as exc:
            logging.debug("HF attempt failed (%s): %s", dataset_name, exc)

    logging.info("No HuggingFace dataset found — falling back to Google Drive download.")
    data_dir = Path(__file__).resolve().parent.parent / "data" / "musique"
    return _download_musique_gdrive(data_dir)


def get_passages(example: Dict) -> List[str]:
    """Extract paragraph texts from a MuSiQue example safely."""
    paragraphs = example.get("paragraphs", [])
    passages: List[str] = []

    for paragraph in paragraphs:
        if isinstance(paragraph, dict):
            text = paragraph.get("paragraph_text") or paragraph.get("text") or ""
        else:
            text = str(paragraph)
        if text and text.strip():
            passages.append(text.strip())

    return passages


def infer_hop_depth(example: Dict) -> int:
    """Infer hop depth by counting supporting paragraphs when available."""
    paragraphs = example.get("paragraphs", [])
    supporting = 0
    for paragraph in paragraphs:
        if isinstance(paragraph, dict) and paragraph.get("is_supporting"):
            supporting += 1

    if supporting > 0:
        return supporting
    return 0


def filter_answerable(examples: List[Dict]) -> List[Dict]:
    """Filter to answerable examples if the field exists."""
    has_flag = any("answerable" in example for example in examples)
    if not has_flag:
        return examples
    return [example for example in examples if bool(example.get("answerable", True))]


def plot_f1_by_hop(metrics: Dict, output_path: Path) -> None:
    """Plot F1 by inferred hop depth."""
    f1_by_hops = metrics.get("f1_by_hops", {})
    if not f1_by_hops:
        logging.warning("No f1_by_hops values found; skipping F1 plot.")
        return

    hops = sorted(int(k) for k in f1_by_hops.keys())
    values = [float(f1_by_hops[h]) for h in hops]

    plt.figure(figsize=(8, 5))
    plt.plot(hops, values, marker="o", linewidth=2)
    plt.xlabel("Hop Depth")
    plt.ylabel("F1")
    plt.title("Phase 1: F1 vs Hop Depth")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_context_growth(metrics: Dict, output_path: Path) -> None:
    """Plot average context tokens across IRCoT generation hops."""
    tokens_by_hop = metrics.get("context_tokens_by_hop", {})
    if not tokens_by_hop:
        logging.warning("No context_tokens_by_hop values found; skipping context plot.")
        return

    hops = sorted(int(k) for k in tokens_by_hop.keys())
    averages = [sum(tokens_by_hop[h]) / len(tokens_by_hop[h]) for h in hops]

    plt.figure(figsize=(8, 5))
    plt.plot(hops, averages, marker="o", linewidth=2)
    plt.xlabel("IRCoT Generation Hop")
    plt.ylabel("Average Context Tokens")
    plt.title("Phase 1: Context Growth per Hop")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_phase1(config: SPIREConfig) -> Dict:
    """Execute the full Phase 1 pipeline and return run artifact payload."""
    logging.info("Loading model: %s", config.model_name)
    model_manager = ModelManager(config)

    examples = load_musique_validation()
    examples = filter_answerable(examples)
    examples = examples[: config.num_examples]
    logging.info("Running on %d examples", len(examples))

    evaluator = Evaluator()
    results: List[Dict] = []
    gold_answers: List[str] = []
    hop_depths: List[int] = []

    for example in tqdm(examples, desc="Phase 1 IRCoT"):
        question = example.get("question", "")
        gold_answer = example.get("answer", "")
        passages = get_passages(example)

        if not question or not passages:
            continue

        retriever = BM25Retriever(passages=passages)
        ircot = IRCoTLoop(model=model_manager, retriever=retriever, config=config)
        result = ircot.run(question=question)

        results.append(result)
        gold_answers.append(gold_answer)
        hop_depths.append(infer_hop_depth(example))

    metrics = evaluator.evaluate_results(results=results, gold_answers=gold_answers, hop_depths=hop_depths)

    return {
        "config": vars(config),
        "metrics": metrics,
        "results": results,
        "gold_answers": gold_answers,
        "hop_depths": hop_depths,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": config.model_name,
        "phase": 1,
    }


def save_outputs(payload: Dict, output_dir: Path) -> None:
    """Persist JSON run artifact and required Phase 1 plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json_path = output_dir / f"run_{timestamp}.json"
    with run_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metrics = payload.get("metrics", {})
    plot_f1_by_hop(metrics, output_dir / f"f1_by_hop_{timestamp}.png")
    plot_context_growth(metrics, output_dir / f"context_tokens_by_hop_{timestamp}.png")

    logging.info("Saved run artifact: %s", run_json_path)


def main() -> None:
    """Run Phase 1 experiment entrypoint."""
    setup_logging()
    config = SPIREConfig()

    payload = run_phase1(config)
    save_outputs(payload, config.output_path())

    metrics = payload.get("metrics", {})
    logging.info("Overall F1: %.4f", metrics.get("overall_f1", 0.0))
    logging.info("Overall EM: %.4f", metrics.get("overall_em", 0.0))


if __name__ == "__main__":
    main()
