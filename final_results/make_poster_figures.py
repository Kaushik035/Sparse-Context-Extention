from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "poster_assets" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def pick_latest(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern}")
    return matches[-1]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def average_series(series: dict) -> dict[int, float]:
    hop_map: dict[int, float] = {}
    for hop_key, values in series.items():
        try:
            hop = int(hop_key)
        except Exception:
            continue
        hop_map[hop] = average(values if isinstance(values, list) else [values])
    return hop_map


def effective_attended_tokens(
    baseline: str,
    metrics: dict,
    base_config: dict,
) -> dict[int, float]:
    context_tokens = metrics.get("context_tokens_by_hop", {})
    real_avg_by_hop = average_series(context_tokens)
    if baseline in {"B5", "B6", "B7"}:
        hash_budget = 0 if baseline == "B5" else int(base_config.get("hash_budget", 0))
        attended = int(base_config.get("sink_size", 0)) + int(base_config.get("local_window", 0)) + hash_budget
        return {hop: min(real_avg, float(attended)) for hop, real_avg in real_avg_by_hop.items()}

    return real_avg_by_hop


def read_metrics() -> tuple[dict, dict]:
    phase2 = load_json(pick_latest("results/phase2/run_*.json"))
    phase3 = load_json(pick_latest("results/phase3/run_*.json"))
    return phase2, phase3


def build_results_figure(phase2: dict, phase3: dict) -> Path:
    metrics2 = phase2["metrics_by_config"]
    metrics3 = phase3["metrics_by_config"]
    base2 = phase2.get("base_config", {})
    base3 = phase3.get("base_config", {})

    # Measured baselines from the saved runs.
    labels = ["B1", "B2", "B3", "B5", "B6", "B7"]
    friendly = [
        "Retrieve-Once",
        "Dense IRCoT",
        "Truncate",
        "Sink+Local",
        "SPIRE Full",
        "SPIRE + Attn",
    ]

    f1 = [
        metrics3["retrieve_once"]["overall_f1"],
        metrics2["dense"]["overall_f1"],
        metrics3["truncate_b3"]["overall_f1"],
        metrics2["spire_b5"]["overall_f1"],
        metrics2["spire_b6"]["overall_f1"],
        metrics3["spire_attn"]["overall_f1"],
    ]
    em = [
        metrics3["retrieve_once"]["overall_em"],
        metrics2["dense"]["overall_em"],
        metrics3["truncate_b3"]["overall_em"],
        metrics2["spire_b5"]["overall_em"],
        metrics2["spire_b6"]["overall_em"],
        metrics3["spire_attn"]["overall_em"],
    ]

    right_panel_sources = {
        "B1": (metrics3["retrieve_once"], base3),
        "B2": (metrics3["dense"], base3),
        "B3": (metrics3["truncate_b3"], base3),
        "B5": (metrics2["spire_b5"], base2),
        "B6": (metrics2["spire_b6"], base2),
        "B7": (metrics3["spire_attn"], base3),
    }

    effective_by_baseline = {
        baseline: effective_attended_tokens(baseline, metrics, base_cfg)
        for baseline, (metrics, base_cfg) in right_panel_sources.items()
    }
    ctx_hops = sorted({hop for hop_map in effective_by_baseline.values() for hop in hop_map})

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")

    # Left: grouped horizontal bars for easier reading in a poster.
    ax = axes[0]
    y = list(range(len(labels)))
    bar_h = 0.34
    ax.barh([i + bar_h / 2 for i in y], f1, height=bar_h, color="#2563eb", label="F1")
    ax.barh([i - bar_h / 2 for i in y], em, height=bar_h, color="#f97316", label="EM")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  {name}" for lab, name in zip(labels, friendly)])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Measured benchmark results")
    ax.set_xlabel("Score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=False)

    for idx, (f1_val, em_val) in enumerate(zip(f1, em)):
        ax.text(f1_val + 0.008, idx + bar_h / 2, f"{f1_val:.3f}", va="center", ha="left", fontsize=10)
        ax.text(em_val + 0.008, idx - bar_h / 2, f"{em_val:.3f}", va="center", ha="left", fontsize=10)

    # Right: effective attended tokens per hop.
    ax = axes[1]
    style_map = {
        "B1": ("#8b5cf6", "o", "-", "B1 Retrieve-Once"),
        "B2": ("#2563eb", "o", "-", "B2 Dense IRCoT"),
        "B3": ("#dc2626", "v", "--", "B3 Truncate"),
        "B5": ("#ea580c", "s", "-", "B5 Sink+Local"),
        "B6": ("#16a34a", "^", "-", "B6 SPIRE Full"),
        "B7": ("#0f766e", "D", "-.", "B7 SPIRE + Attn"),
    }
    for baseline in ["B1", "B2", "B3", "B5", "B6", "B7"]:
        values_map = effective_by_baseline[baseline]
        y_vals = [values_map.get(h) for h in ctx_hops]
        color, marker, linestyle, label = style_map[baseline]
        ax.plot(ctx_hops, y_vals, marker=marker, linewidth=2.75, linestyle=linestyle, color=color, label=label)
        for hop, val in zip(ctx_hops, y_vals):
            if val is not None:
                ax.annotate(f"{val/1000:.2f}K", (hop, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("Effective attended tokens per hop")
    ax.set_xlabel("IRCoT generation hop")
    ax.set_ylabel("Effective attended tokens")
    ax.set_xticks(ctx_hops)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1000:.2f}K"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=False)

    fig.suptitle("SPIRE poster summary: retrieval quality and context pressure", y=0.98, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / "poster_results_summary.svg"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT_DIR / "poster_results_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    phase2, phase3 = read_metrics()
    out = build_results_figure(phase2, phase3)
    print(out)


if __name__ == "__main__":
    main()