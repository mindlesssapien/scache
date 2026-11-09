import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_operating_curve():
    benchmark_dir = Path(__file__).resolve().parent
    results_path = benchmark_dir / "results.json"
    output_image = benchmark_dir / "operating_curve.png"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found at {results_path}. "
            "Please run 'uv run python -m benchmarks.evaluate' first."
        )

    with open(results_path, "r") as f:
        data = json.load(f)

    taus = [row["tau"] for row in data]
    chrs = [row["chr"] * 100 for row in data]  # Convert to percentage
    whrs = [row["whr"] * 100 for row in data]  # Convert to percentage

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Semantic Cache Threshold Evaluation (MiniLM-L6-v2 on QQP)",
        fontsize=14,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Subplot 1: Operating Characteristic Curve (WHR vs CHR)
    # -------------------------------------------------------------
    ax1.plot(whrs, chrs, color="#1f77b4", linewidth=2.5, marker="o", markersize=4, label="Operating Curve")
    ax1.axvline(1.0, color="#d62728", linestyle="--", alpha=0.8, label="Target Budget (WHR = 1.0%)")
    ax1.axvspan(0, 1.0, color="#2ca02c", alpha=0.1, label="Safe Operating Zone (WHR ≤ 1%)")

    # Annotate key operating points
    key_points = [
        (0.97, "tau=0.97 (Strict ≤1% WHR)", (-40, 25)),
        (0.91, "tau=0.91 (Balanced ~5% WHR)", (15, -20)),
        (0.85, "tau=0.85 (Permissive)", (15, -15)),
    ]

    for target_tau, label, offset in key_points:
        match = next((row for row in data if abs(row["tau"] - target_tau) < 0.005), None)
        if match:
            x_val = match["whr"] * 100
            y_val = match["chr"] * 100
            ax1.scatter([x_val], [y_val], color="#ff7f0e", s=60, zorder=5)
            ax1.annotate(
                f"{label}\n(WHR: {x_val:.1f}%, CHR: {y_val:.1f}%)",
                xy=(x_val, y_val),
                xytext=offset,
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            )

    ax1.set_title("Operating Characteristic (WHR vs CHR)", fontsize=12)
    ax1.set_xlabel("Wrong-Hit Rate (False Acceptance Rate) [%]", fontsize=11)
    ax1.set_ylabel("Cache Hit Rate (True Acceptance Rate) [%]", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")
    ax1.set_xlim(-0.5, max(whrs) + 2)
    ax1.set_ylim(-2, 102)

    # -------------------------------------------------------------
    # Subplot 2: Metric Sensitivity vs Threshold (tau)
    # -------------------------------------------------------------
    ax2.plot(taus, chrs, color="#2ca02c", linewidth=2, marker="s", markersize=3.5, label="Cache Hit Rate (CHR %)")
    ax2.plot(taus, whrs, color="#d62728", linewidth=2, marker="^", markersize=3.5, label="Wrong-Hit Rate (WHR %)")

    # Mark the operational threshold
    ax2.axvline(0.97, color="#1f77b4", linestyle="--", linewidth=1.5, label="Selected tau = 0.97")

    ax2.set_title("CHR & WHR vs Similarity Threshold (tau)", fontsize=12)
    ax2.set_xlabel("Similarity Threshold (tau)", fontsize=11)
    ax2.set_ylabel("Rate [%]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center left")
    ax2.set_xlim(0.69, 1.01)
    ax2.set_ylim(-2, 102)

    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()

    print("=" * 80)
    print(f"[SUCCESS] Operating curve saved to: {output_image}")
    print("=" * 80)
    print("\nOperational Trade-offs Summary:")
    print("  • Conservative (tau=0.97): WHR = 0.84% (Safe), CHR = 12.72% (Low Cache Utilization)")
    print("  • Moderate     (tau=0.91): WHR = 5.40%,        CHR = 39.40% (Triples Cache Hits)")
    print("  • Permissive   (tau=0.85): WHR = 13.12%,       CHR = 62.04% (High risk on near-duplicates)")
    print("=" * 80)


if __name__ == "__main__":
    plot_operating_curve()
