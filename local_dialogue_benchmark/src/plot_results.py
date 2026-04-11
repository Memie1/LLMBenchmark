from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results" / "benchmark_results.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def main():
    if not RESULTS_FILE.exists():
        print("No results file found. Run benchmark_runner first.")
        return

    PLOTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(RESULTS_FILE)

    # 1. Average response time 
    plt.figure()
    df.groupby("scenario_title")["response_time_ms"].mean().plot(kind="bar")
    plt.title("Average Response Time per Scenario")
    plt.ylabel("ms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_response_time.png")
    plt.close()

    # 2. Pass rate per scenario
    plt.figure()
    df.groupby("scenario_title")["passed_basic_checks"].mean().plot(kind="bar")
    plt.title("Pass Rate per Scenario")
    plt.ylabel("Pass Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pass_rate.png")
    plt.close()

    # 3. Speed vs reliability 
    summary = df.groupby("preset").agg({
        "response_time_ms": "mean",
        "passed_basic_checks": "mean"
    })

    plt.figure()
    plt.scatter(summary["response_time_ms"], summary["passed_basic_checks"])

    for preset, row in summary.iterrows():
        plt.text(row["response_time_ms"], row["passed_basic_checks"], preset)

    plt.xlabel("Avg Response Time (ms)")
    plt.ylabel("Pass Rate")
    plt.title("Speed vs Reliability")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "speed_vs_reliability.png")
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()